import tensorflow as tf
import json

from google.cloud import storage

import glob
import os


# GCP Helpers


def gcp_path(path, bucket_name):
    return "gs://{}/{}".format(bucket_name, path)


def gcp_exists(path, bucket_name):
    return tf.io.gfile.exists(gcp_path(path, bucket_name))


def gcp_ls(path, bucket_name):
    client = storage.Client()
    return client.list_blobs(bucket_name, prefix=path)


def gcp_cp_dir_contents(gcp_dir, local_dir, bucket_name):
    try:
        os.mkdir(local_dir)
    except OSError:
        print("Creation of the directory %s failed" % local_dir)
    else:
        print("Successfully created the directory %s " % local_dir)

    for blob in gcp_ls(gcp_dir, bucket_name):
        download_path = local_dir + blob.name.split("/")[-1]
        blob.download_to_filename(download_path)


def cp_local_directory_to_gcp(local_path, gcp_path, bucket_name):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + "/**"):
        if not os.path.isfile(local_file):
            continue
        remote_path = os.path.join(gcp_path, local_file[1 + len(local_path) :])
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


def read_mrqa(path, labels, max_length=None):
    with tf.io.gfile.GFile(path, "rb") as f:
        data = [json.loads(jline) for jline in f.readlines()]

    contexts = []
    questions = []
    answers = []
    for data_ex in data[1:]:

        context = data_ex["context"]

        for qa in data_ex["qas"]:
            question = qa["question"]

            if labels:

                # sometimes many acceptable spans, just pick
                # one in the middle and hope for the best.
                span = qa["detected_answers"][
                    len(qa["detected_answers"]) // 2
                ]["char_spans"]

                answer_start, answer_end = span[0]

            else:

                # Signal pseudo labeling by
                # setting negative spans
                answer_start = -1
                answer_end = -1

            contexts.append(context)
            questions.append(question)
            answers.append(
                {
                    "answers": qa["answers"],
                    "answer_start": answer_start,
                    "answer_end": answer_end,
                }
            )
            if max_length is not None and len(answers) >= max_length:
                return contexts, questions, answers

    return contexts, questions, answers


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):

        token_start = answers[i]["answer_start"]
        token_end = answers[i]["answer_end"]

        if token_start != -1:
            start_positions.append(encodings.char_to_token(i, token_start))
            end_positions.append(encodings.char_to_token(i, token_end))
        else:
            start_positions.append(-1)
            end_positions.append(-1)

        # if None, the answer passage has been truncated
        if start_positions[-1] is None or end_positions[-1] is None:
            start_positions[-1] = 0
            end_positions[-1] = 0

    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )


def parse_mrqa_pseudo(path, tokenizer):
    contexts, questions, answers = read_mrqa(
        path, labels=False, max_length=None
    )

    encodings = tokenizer(contexts, questions, truncation=True, padding=True)

    # Answers need to be of 'rectangular size' before we feed them into
    # tf.data.Dataset.from_tensor_slices. Thus, we need to jump thorugh
    # some hoops here.

    # Join all possible answers to a question into a single string delimited by
    # a non vocabulary character
    answer_list = ["\x00".join(answer["answers"]) for answer in answers]

    # Padd to rectngular size, whitespace will be removed later anyway
    max_answer_length = len(max(answer_list, key=len))
    padded_answers = [
        answer.ljust(max_answer_length) for answer in answer_list
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                key: encodings[key]
                for key in ["input_ids", "attention_mask", "token_type_ids"]
            },
            padded_answers,
        )
    )
    return dataset, len(answers)


def parse_mrqa(path, tokenizer, max_length=None):
    contexts, questions, answers = read_mrqa(
        path, labels=True, max_length=max_length
    )

    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers)

    # Answers need to be of 'rectangular size' before we feed them into
    # tf.data.Dataset.from_tensor_slices. Thus, we need to jump thorugh
    # some hoops here.

    # Join all possible answers to a question into a single string delimited by
    # a non vocabulary character
    answer_list = ["\x00".join(answer["answers"]) for answer in answers]

    # Padd to rectngular size, whitespace will be removed later anyway
    max_answer_length = len(max(answer_list, key=len))
    padded_answers = [
        answer.ljust(max_answer_length) for answer in answer_list
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                key: encodings[key]
                for key in ["input_ids", "attention_mask", "token_type_ids"]
            },
            (
                encodings["start_positions"],
                encodings["end_positions"],
            ),
            padded_answers,
        )
    )
    return dataset, len(answers)


def get_preds(model, tokenizer, path):

    with tf.io.gfile.GFile(path, "rb") as f:
        data = [json.loads(jline) for jline in f.readlines()]

    too_long = 0
    preds = {}

    for data_record in data:
        text = data_record["context"]

        for qa in data_record["qas"]:
            question = qa["question"]
            qid = qa["qid"]

            input_dict = tokenizer(question, text, return_tensors="tf")
            context_tokens = input_dict["input_ids"].numpy()[0]
            if context_tokens.size < 512:

                model_outputs = model(input_dict)
                start_logits = model_outputs[0]
                end_logits = model_outputs[1]

                # Has batch dimmension
                model_answer = model.batch_decode_answers(
                    input_dict, start_logits, end_logits
                )

                preds[qid] = model_answer

            else:
                too_long += 1

    print(f"Excluded {too_long} questions for having too long contexts.")

    return preds


def Kfold(dataset, num_folds, length=None):

    for i in range(num_folds):

        is_train = lambda x, y: x % num_folds != i
        is_eval = lambda x, y: x % num_folds == i
        recover = lambda x, y: y

        if length is not None:
            yield (
                dataset.take(length).enumerate().filter(is_train).map(recover),
                dataset.take(length).enumerate().filter(is_eval).map(recover),
            )
            # Hacky way to not repeat the squad data
            dataset = dataset.skip(length)
        else:
            yield (
                dataset.enumerate().filter(is_train).map(recover),
                dataset.enumerate().filter(is_eval).map(recover),
            )
