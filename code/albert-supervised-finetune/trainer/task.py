import os
import json

from . import util
from . import parse

import tensorflow as tf
import numpy as np

from transformers import TFAlbertForQuestionAnswering
from transformers import AlbertTokenizerFast

from tqdm.keras import TqdmCallback


def train_and_evaluate(args):
    """
    Main method of the entire job. Loads data and BERT and performs training and
    evaluation as specified by the argument in args.

    Parameters:
        args (dictionary) : A dictionary of arguments.
    """

    # LOADING ALBERT #

    # First we Load Albert
    model_string = "twmkn9/albert-base-v2-squad2"

    if args.remote:

        data_path = util.gcp_path(args.data_path, args.bucket_name)
        test_data_path = util.gcp_path(args.test_data_path, args.bucket_name)

        # If we are using TPU or multiple GPU's then we need to define a
        # tf.distribute.Strategy. This is basically what allows us to use
        # multiple machines in parallell.
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

        data_path = args.data_path
        test_data_path = args.test_data_path

        # This is needed for local training sometimes idk why
        TF_CONFIG = os.environ.get("TF_CONFIG")
        if TF_CONFIG:
            os.environ.pop("TF_CONFIG")

    # XLA stands for Accelarated Linnear Algebra. This is basically just
    # optimzation. It does JIT for linnear algebra stuff and also reduces the
    # amount of copying done by TF
    if args.xla:
        tf.config.optimizer.set_jit(True)

    # Create model. If we are using a strategy, then the model needs to be
    # created within its scope.
    if args.distributed or args.tpu:
        with strategy.scope():
            albert = TFAlbertForQuestionAnswering.from_pretrained(
                model_string, from_pt=True
            )
            tokenizer = AlbertTokenizerFast.from_pretrained(
                model_string, use_fast=True
            )

    else:
        albert = TFAlbertForQuestionAnswering.from_pretrained(
            model_string, from_pt=True
        )
        tokenizer = AlbertTokenizerFast.from_pretrained(
            model_string, use_fast=True
        )

    # PREPARING THE DATASET #

    # Send one batch to each machine/core in our strategy if we're training
    # ditstributed. Hence the 'global' batch_size is our batch_size_per_machine
    # * num_machines
    if args.distributed:
        global_batch_size = args.batch_size * strategy.num_replicas_in_sync
        print(
            f"Global batch size: {global_batch_size}. Num replicas in sync: {strategy.num_replicas_in_sync}."
        )
    else:
        global_batch_size = args.batch_size

    preds = util.get_preds(albert, tokenizer, test_data_path)

    with tf.io.gfile.GFile(args.job_dir + "/test_preds_pre.jsonl", "w+") as f:
        json.dump(preds, f)

    dataset = util.mrqa_tf_from_jsonl(
        path=data_path, tokenizer=tokenizer
    ).shuffle(2000)

    # train-local is just for debuging
    if not args.remote:
        dataset = dataset.take(3)

    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    if args.amp:
        # Automatic Mixed Precision.
        #
        # This is just optimzation, do calculations in float16 but store
        # parameters as float32.
        #
        # The following is the default way to do it:
        #
        # from tensorflow.keras.mixed_precision import experimental as mixed_precision
        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_policy(policy)
        #
        # but it's not compatibale with transformers, see
        # https://github.com/huggingface/transformers/issues/3320
        # and
        # https://github.com/amaiya/ktrain/issues/126

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

    albert.compile(optimizer=optimizer, loss=loss)

    if args.kfold_eval:

        # Save untrained weights for reset between folds in cross validation
        weight_reset = albert.get_weights()  # TODO randomize reset

        history = []

        # Train model under Kfold cross validation
        for kfold_iter, (train_fold, val_fold) in enumerate(
            util.Kfold(dataset, args.num_folds)
        ):
            # Reset weights between Kfolds iterations
            albert.set_weights(weight_reset)

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
                albert.fit(
                    train_fold,
                    validation_data=val_fold,
                    epochs=args.num_epochs,
                    verbose=1,
                    callbacks=[tensorboard_cb, learning_rate_cb, PrintLR()],
                )
            )

        # Compute average of metrics over Kfold iterations
        summary_writer = tf.summary.create_file_writer(
            os.path.join(args.job_dir, "keras_tensorboard", "kfold_avg")
        )

        # This is the only way that I could quickly find that adds an entire
        # Graph to Tensorboard. Could definetly be prettier
        with summary_writer.as_default():

            for metric in history[0].history.keys():
                metric_avg = np.zeros(len(history[0].history[metric]))

                for history_iter in history:
                    metric_avg += np.array(history_iter.history[metric])

                metric_avg /= args.num_folds

                for step, val in enumerate(metric_avg):
                    tf.summary.scalar(metric, data=val, step=step)

        # Do one final training with full dataset and extract predictions on
        # testset
        albert.set_weights(weight_reset)

    albert.fit(
        dataset.batch(args.batch_size, drop_remainder=True),
        epochs=args.num_epochs,
        verbose=1,
        callbacks=[learning_rate_cb, PrintLR()],
    )

    preds = util.get_preds(albert, tokenizer, test_data_path)

    with tf.io.gfile.GFile(args.job_dir + "/test_preds.jsonl", "w+") as f:
        json.dump(preds, f)


if __name__ == "__main__":
    args = parse.get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
