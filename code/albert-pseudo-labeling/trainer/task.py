from . import util
from . import parse
from . import model
from . import metrics

import tensorflow as tf
import numpy as np

from transformers import TFAlbertForQuestionAnswering
from transformers import AlbertTokenizerFast

import json
import os


def train_and_evaluate(args):

    # Which pretrained model to use, needs to be an albert (probably)
    model_string = "twmkn9/albert-base-v2-squad2"

    if args.remote:

        bio_data_path = util.gcp_path(args.bio_data_path, args.bucket_name)
        squad_data_path = util.gcp_path(args.squad_data_path, args.bucket_name)
        test_data_path = util.gcp_path(args.test_data_path, args.bucket_name)

        if args.tpu:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
            print("All devices: ", tf.config.list_logical_devices("TPU"))

        elif args.distributed:
            strategy = tf.distribute.MirroredStrategy()
            print("All devices: ", tf.config.list_logical_devices("GPU"))

    else:

        bio_data_path = args.bio_data_path
        squad_data_path = args.squad_data_path
        test_data_path = args.test_data_path

        # This is needed for local training sometimes idk why
        TF_CONFIG = os.environ.get("TF_CONFIG")
        if TF_CONFIG:
            os.environ.pop("TF_CONFIG")

    if args.xla:
        tf.config.optimizer.set_jit(True)

    if args.distributed or args.tpu:
        with strategy.scope():
            albert = TFAlbertForQuestionAnswering.from_pretrained(
                model_string, from_pt=True
            )
            tokenizer = AlbertTokenizerFast.from_pretrained(
                model_string, use_fast=True
            )
            pseudo_albert = model.AlbertQA(
                albert, tokenizer, args.pseudo_weight
            )

    else:
        albert = TFAlbertForQuestionAnswering.from_pretrained(
            model_string, from_pt=True
        )
        tokenizer = AlbertTokenizerFast.from_pretrained(
            model_string, use_fast=True
        )
        pseudo_albert = model.AlbertQA(albert, tokenizer, args.pseudo_weight)

    if args.distributed:
        global_batch_size = args.batch_size * strategy.num_replicas_in_sync
        print(
            f"Global batch size: {global_batch_size}. Num replicas in sync: {strategy.num_replicas_in_sync}."
        )
    else:
        global_batch_size = args.batch_size

    bio_dataset, bio_length = util.parse_mrqa_pseudo(
        path=bio_data_path, tokenizer=tokenizer
    )
    print(bio_length)

    # training locally is just for debuging, don't bite off more than my laptop
    # can chew
    if not args.remote:
        bio_length = 9

    # With some hacks in the util.Kfolds implementation we can avoid
    # repeating squad data over epoch. Making it so that new squad
    # data is encountered on each epcoh. Thats why we multiply with num_epochs
    # below.
    squad_dataset, _ = util.parse_mrqa(
        path=squad_data_path,
        tokenizer=tokenizer,
        max_length=(bio_length * args.num_epochs),
    )

    bio_dataset = bio_dataset.shuffle(bio_length).repeat(-1)
    squad_dataset = squad_dataset.shuffle(bio_length).repeat(-1)

    dataset = tf.data.Dataset.zip((bio_dataset, squad_dataset))

    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Automatic Mixed Precision.
    if args.amp:
        tf.config.optimizer.set_experimental_options(
            {"auto_mixed_precision": True}
        )
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            optimizer, "dynamic"
        )

    def lr_warmup(epoch, lr):
        if epoch < args.num_epochs / 3:
            return args.learning_rate * (
                (epoch + 1.0) / ((args.num_epochs + 1.0) / 3.0)
            )
        else:
            return args.learning_rate

    learning_rate_cb = tf.keras.callbacks.LearningRateScheduler(
        lr_warmup, verbose=0
    )

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(
                "\nLearning rate for epoch {} is {}".format(
                    epoch + 1, self.model.optimizer.lr.numpy()
                )
            )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    eval_metrics = [
        metrics.MrqaExactMatch(name="pseudo_exact_match"),
        metrics.MrqaExactMatch(name="labeled_exact_match"),
        metrics.MrqaTokenF1Mean(name="pseudo_token_f1_mean"),
        metrics.MrqaTokenF1Mean(name="labeled_token_f1_mean"),
    ]

    pseudo_albert.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)

    # This is needed since we are using metrics in an unconventional way,
    # updating them with different data in test_step
    pseudo_albert.compiled_metrics._build("", ["", ""])

    if args.kfold_eval:  # Do kfolds evaluation?

        # Save untrained weights for reset between folds in cross validation
        weight_reset = pseudo_albert.get_weights()  # TODO randomize reset?

        history = []

        # Train model under Kfold cross validation
        for kfold_iter, (train_fold, val_fold) in enumerate(
            util.Kfold(dataset, num_folds=args.num_folds, length=bio_length)
        ):
            # Reset weights between Kfolds iterations
            pseudo_albert.set_weights(weight_reset)

            tensorboard_cb = tf.keras.callbacks.TensorBoard(
                os.path.join(
                    args.job_dir, "tensorboard", f"Kfold_iter_{kfold_iter}"
                ),
                update_freq="epoch",
                write_graph=False,
                write_images=False,
                histogram_freq=0,
            )

            train_fold = train_fold.batch(args.batch_size, drop_remainder=True)
            val_fold = val_fold.batch(args.batch_size, drop_remainder=True)

            history.append(
                pseudo_albert.fit(
                    train_fold,
                    validation_data=val_fold,
                    epochs=args.num_epochs,
                    verbose=1,
                    callbacks=[tensorboard_cb, learning_rate_cb, PrintLR()],
                )
            )

        # Compute average of metrics over Kfold iterations
        summary_writer = tf.summary.create_file_writer(
            os.path.join(args.job_dir, "tensorboard", "kfold_avg")
        )

        # Add average of all plots to tensorboard
        with summary_writer.as_default():

            for metric in history[0].history.keys():
                metric_avg = np.zeros(len(history[0].history[metric]))

                for history_iter in history:
                    metric_avg += np.array(history_iter.history[metric])

                metric_avg /= args.num_folds

                for step, val in enumerate(metric_avg):
                    tf.summary.scalar(metric, data=val, step=step)

        pseudo_albert.set_weights(weight_reset)

    # Do one final training with full dataset and extract predictions on
    # testset

    pseudo_albert.fit(
        dataset.batch(args.batch_size, drop_remainder=True),
        epochs=args.num_epochs,
        steps_per_epoch=bio_length,
        verbose=1,
        callbacks=[learning_rate_cb, PrintLR()],
    )

    # Retrieve test preds so we can evaluate the model with the MRQA 2019
    # task evaluate code (this minimizes chanses of eval bugs).
    preds = util.get_preds(pseudo_albert, tokenizer, test_data_path)

    with tf.io.gfile.GFile(args.job_dir + "/test_preds.jsonl", "w+") as f:
        json.dump(preds, f)

    pseudo_albert.albert.save_pretrained("/tmp/saved_albert")

    model_dir = "/models/" + args.job_dir.split("/")[-1]
    util.cp_local_directory_to_gcp(
        "/tmp/saved_albert", model_dir, args.bucket_name
    )


if __name__ == "__main__":
    args = parse.get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
