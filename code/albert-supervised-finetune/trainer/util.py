import tensorflow as tf
import json

from google.cloud import storage

import os

# GCP Helpers


def gcp_path(path, bucket_name):
    """
    This function accepts a google cloud platform path and bucket and returns a
    URI to it.

    Parameters:
        path (str): A google cloud platform path
        bucket_name: A google cloud platform bucket name

    returns:
        A google cloud platform URI to the specified storage path and bucket

    """

    return "gs://{}/{}".format(bucket_name, path)


def gcp_exists(path, bucket_name):
    """
    This function accepts a google cloud platform path and bucket and returns
    and checks to see if it exists.


    Parameters:
        path (str): A google cloud platform path
        bucket_name: A google cloud platform bucket name

    returns:
        A boolean that signifies the existance of specified storage path and
        bucket
    """

    return tf.io.gfile.exists(gcp_path(path, bucket_name))


def gcp_ls(path, bucket_name):
    """
    This function accepts a google cloud platform directory path and bucket and returns
    a list of google.cloud.storage.blob.Blob objects that reside under the
    passed path


    Parameters:
        path (str): A google cloud platform directory path
        bucket_name: A google cloud platform bucket name

    returns:
        A list of google.cloud.storage.blob.Blob objects that reside under the
        passed path
    """

    client = storage.Client()
    return client.list_blobs(bucket_name, prefix=path)


def gcp_cp_dir_contents(gcp_dir, local_dir, bucket_name):
    """
    This function accepts a google cloud platform directory path, a local path
    and bucket name and copies all of the contents inside the google cloud
    platform directory to the local directory


    Parameters:
        gcp_dir (str): A google cloud platform directory path
        local_dir (str): A local directory path
        bucket_name: A google cloud platform bucket name

    """

    try:
        os.mkdir(local_dir)
    except OSError:
        print("Creation of the directory %s failed" % local_dir)
    else:
        print("Successfully created the directory %s " % local_dir)

    for blob in gcp_ls(gcp_dir, bucket_name):
        download_path = local_dir + blob.name.split("/")[-1]
        blob.download_to_filename(download_path)


def read_mrqa(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        data = [json.loads(jline) for jline in f.readlines()]

    contexts = []
    questions = []
    answers = []
    for data_ex in data[1:]:

        context = data_ex["context"]

        for qa in data_ex["qas"]:
            question = qa["question"]

            span = qa["detected_answers"][len(qa["detected_answers"]) // 2][
                "char_spans"
            ]
            answer_start, answer_end = span[0]
            if answer_end == 0:
                continue

            contexts.append(context)
            questions.append(question)
            answers.append(
                {"answer_start": answer_start, "answer_end": answer_end}
            )

    return contexts, questions, answers


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(
            encodings.char_to_token(i, answers[i]["answer_start"])
        )
        end_positions.append(
            encodings.char_to_token(i, answers[i]["answer_end"])
        )

        # if None, the answer passage has been truncated
        if start_positions[-1] is None or end_positions[-1] is None:
            start_positions[-1] = 0
            end_positions[-1] = 0

    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )


def mrqa_tf_from_jsonl(path, tokenizer):
    contexts, questions, answers = read_mrqa(path)
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers)

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
        )
    )
    return dataset


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

                start_index = tf.math.argmax(start_logits, 1)[0]
                end_index = tf.math.argmax(end_logits, 1)[0]

                model_answer = tokenizer.decode(
                    context_tokens[start_index:end_index + 1]
                )

                preds[qid] = model_answer

            else:
                too_long += 1

    print(f"Excluded {too_long} questions for having too long contexts.")

    return preds


def Kfold(dataset, num_folds):

    for i in range(num_folds):

        is_train = lambda x, y: x % num_folds != i
        is_eval = lambda x, y: x % num_folds == i
        recover = lambda x, y: y

        yield (
            dataset.enumerate().filter(is_train).map(recover),
            dataset.enumerate().filter(is_eval).map(recover),
        )
