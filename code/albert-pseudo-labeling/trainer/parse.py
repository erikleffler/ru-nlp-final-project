import argparse


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="local or GCS location for writing checkpoints and exporting "
        "models",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="number of times to go through the data, default=20",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="number of records to read during each training step, default=128",
    )
    parser.add_argument(
        "--training-steps",
        required=True,
        type=int,
        help="Ammount of training steps",
    )
    parser.add_argument(
        "--learning-rate",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--warmup-steps",
        required=True,
        type=int,
        help="Ammount of warmup steps for learning rate (linnear increase from 0 to learning rate)",
    )
    parser.add_argument(
        "--pseudo-weight",
        required=True,
        type=float,
        help="Loss weight of pseudo labeled examples during training",
    )
    parser.add_argument(
        "--pre-train-mlm-head-steps",
        required=True,
        type=int,
        help="Ammount of training steps for which to train the mlm-head before the main train loop (I.e BERT will be frozen for theese steps))",
    )
    parser.add_argument(
        "--remote",
        type=int,
        default=0,
        help="whether to run in google cloud or local. 1 -> gcp, 0 -> local",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        required=True,
        help="maximum sequence length. Longer sequences will be truncated.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=-1,
        help="maximum sequence length. Longer sequences will be truncated.",
    )
    parser.add_argument(
        "--kfold-eval",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        required=True,
        help="",
    )
    parser.add_argument(
        "--train-steps-per-eval",
        type=int,
        default=100,
        help="",
    )
    parser.add_argument(
        "--squad-data-path",
        type=str,
        required=True,
        help="path to data file with csv data",
    )
    parser.add_argument(
        "--bio-data-path",
        type=str,
        required=True,
        help="path to data file with csv data",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="path to data file with csv data",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        required=True,
        help="name of gcp bucket",
    )
    parser.add_argument(
        "--tpu",
        type=int,
        default=0,
        help="Use tpu?",
    )
    parser.add_argument(
        "--xla",
        type=int,
        default=0,
        help="Use xla?",
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=0,
        help="Use automatic mixed precission?",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="How many train batches to run inbetween each checkpoint save",
    )
    parser.add_argument(
        "--distributed",
        type=int,
        default=0,
        help="Train in a distributed setting? I.e whether to enable tf strategies or not",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
    )
    args, _ = parser.parse_known_args()
    return args
