import tensorflow as tf
import os


class BatchLearningRate(tf.keras.callbacks.Callback):
    def __init__(
        self, warmup_steps, target_learning_rate, global_batch_size, debug=False
    ):
        super(BatchLearningRate, self).__init__()

        self.learning_rate = 0
        self.warmup_steps = warmup_steps
        self.target_learning_rate = target_learning_rate
        self.learning_rate_increment = (
            target_learning_rate * global_batch_size
        ) / warmup_steps
        self.debug = debug

    def __call__(self):
        return self.learning_rate

    def on_train_batch_begin(self, batch, logs=None):
        if self.learning_rate < self.target_learning_rate:
            self.learning_rate = self.learning_rate + self.learning_rate_increment

        if self.debug:
            print()
            print(f"Optimizer learning rate: {self.model.optimizer.lr}")


class BatchCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, path, checkpoint_steps):
        super(BatchCheckpoint, self).__init__()

        self.n_batches = 0
        self.path = path
        self.checkpoint_steps = checkpoint_steps

    def on_train_batch_end(self, batch, logs=None):

        self.n_batches += 1

        if self.n_batches % self.checkpoint_steps == 0:

            save_path = os.path.join(self.path, f"batch_{self.n_batches}")

            print(f"Saving model checkpoint to path: {save_path}")

            # Saving with transformer save_pretrained.
            # https://huggingface.co/transformers/model_sharing.html
            #
            # load model again with:
            #
            #   pretrained_bert = TFBertForMaskedLM.from_pretrained(save_path)

            self.model.save_pretrained(save_path)
