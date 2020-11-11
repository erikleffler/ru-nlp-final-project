import tensorflow as tf

# Imports below is just because we needed to overide an internal function
# to be able to run test_step in eager mode whilst keeping train_step in graph
# mode.
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.python.ops import variables


class AlbertQA(tf.keras.Model):
    def __init__(
        self, model, tokenizer, pseudo_weight, name="albert_for_qa", **kwargs
    ):
        super(AlbertQA, self).__init__(name=name, **kwargs)

        self.albert = model
        self.tokenizer = tokenizer
        self.pseudo_weight = pseudo_weight

    @tf.function
    def call(self, inputs, training=False):
        return self.albert(inputs, training=training)

    @tf.function
    def train_step(self, data):

        pseudo_data, labeled_data = data

        pseudo_x, _ = pseudo_data
        labeled_x, labeled_y, _ = labeled_data

        with tf.GradientTape() as tape:

            pseudo_y_pred = self(pseudo_x, training=True)

            pseudo_y_start = tf.math.argmax(pseudo_y_pred[0], axis=1)
            pseudo_y_end = tf.math.argmax(pseudo_y_pred[1], axis=1)

            # Answer start
            pseudo_loss = self.compiled_loss(
                pseudo_y_start,
                pseudo_y_pred[0],
                regularization_losses=self.losses,
            )
            # Answer end
            pseudo_loss += self.compiled_loss(
                pseudo_y_end,
                pseudo_y_pred[1],
                regularization_losses=self.losses,
            )

            labeled_y_pred = self(labeled_x, training=True)

            # Answer start
            labeled_loss = self.compiled_loss(
                labeled_y[0],
                labeled_y_pred[0],
                regularization_losses=self.losses,
            )
            # Answer end
            labeled_loss += self.compiled_loss(
                labeled_y[1],
                labeled_y_pred[1],
                regularization_losses=self.losses,
            )

            loss = labeled_loss + self.pseudo_weight * pseudo_loss

        trainable_vars = self.albert.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        #        self.compiled_metrics.update_state(y, y_pred)

        return {"pseudo_loss": pseudo_loss, "labeled_loss": labeled_loss}

    def test_step(self, data):
        pseudo_data, labeled_data = data

        pseudo_x, pseudo_answers = pseudo_data
        labeled_x, _, labeled_answers = labeled_data

        # Recreate the answer lists
        pseudo_answers = [
            answers.numpy().decode("utf-8").rstrip().split("\x00")
            for answers in pseudo_answers
        ]

        labeled_answers = [
            answers.numpy().decode("utf-8").rstrip().split("\x00")
            for answers in labeled_answers
        ]

        # Get model predictions
        pseudo_start_logits, pseudo_end_logits = self.albert(
            pseudo_x, training=False
        )

        labeled_start_logits, labeled_end_logits = self.albert(
            labeled_x, training=False
        )

        pseudo_model_answers = self.batch_decode_answers(
            pseudo_x, pseudo_start_logits, pseudo_end_logits
        )

        labeled_model_answers = self.batch_decode_answers(
            labeled_x, labeled_start_logits, labeled_end_logits
        )

        for metric in self.metrics:
            if "pseudo" in metric.name:
                metric.update_state(pseudo_answers, pseudo_model_answers)
            elif "labeled" in metric.name:
                metric.update_state(labeled_answers, labeled_model_answers)

        return {m.name: m.result() for m in self.compiled_metrics.metrics}

    def batch_decode_answers(
        self, batch_input_dict, batch_start_logits, batch_end_logits
    ):

        batch_input_ids = batch_input_dict["input_ids"]
        batch_attention_mask = batch_input_dict["attention_mask"]

        batch_start_probs = tf.nn.softmax(batch_start_logits)
        batch_end_probs = tf.nn.softmax(batch_end_logits)

        answers_tokens = []

        # Need to find most likely span
        # As we need the argmax_{i,j}(start_logits[i] * end_logits[j])
        # where i < j, there is som trickery involved. Idk how to do this
        # in a vectorized manner

        # Outermost loop over entries in batch
        for input_ids, attention_mask, start_probs, end_probs in zip(
            batch_input_ids,
            batch_attention_mask,
            batch_start_probs,
            batch_end_probs,
        ):
            best_span_p = 0
            best_start_i = 0
            best_start_p = 0
            best_span = (0, 1)
            for i in range(1, tf.size(start_probs) - 1):
                start_p = start_probs[i - 1]
                if start_p > best_start_p:
                    best_start_i = i - 1
                    best_start_p = start_p

                end_p = end_probs[i]
                if end_p * best_start_p > best_span_p:
                    best_span = (best_start_i, i)
                    best_span_p = end_p * best_start_p

                if not attention_mask[
                    i
                ]:  # This tensor is 0 for padding tokens
                    break

            answers_tokens.append(input_ids[best_span[0] : best_span[1] + 1])

        # Turn tokens to text and return
        return self.tokenizer.batch_decode(
            answers_tokens, skip_special_tokens=True
        )

    # Everything below can be safely ignored
    # Function overrides that allow self.test_step to be ran in eager mode

    # Overriding this method from tf so that we can do test_step in eager mode
    def make_test_function(self):
        """Creates a function that executes one step of evaluation.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.evaluate` and `Model.test_on_batch`.
        Typically, this method directly controls `tf.function` and
        `tf.distribute.Strategy` settings, and delegates the actual evaluation
        logic to `Model.test_step`.
        This function is cached the first time `Model.evaluate` or
        `Model.test_on_batch` is called. The cache is cleared whenever
        `Model.compile` is called.
        Returns:
          Function. The function created by this method should accept a
          `tf.data.Iterator`, and return a `dict` containing values that will
          be passed to `tf.keras.Callbacks.on_test_batch_end`.
        """
        if self.test_function is not None:
            return self.test_function

        def test_function(iterator):
            data = next(iterator)
            return self.test_step(data)

        # CHANGED HERE
        # Removed call to def_function. I think it turns test_funciton
        # into a tf.function, which is what we need to avoid

        self.test_function = test_function
        return self.test_function


def _minimum_control_deps(outputs):
    """Returns the minimum control dependencies to ensure step succeeded."""
    if context.executing_eagerly():
        return []  # Control dependencies not needed.
    outputs = nest.flatten(outputs, expand_composites=True)
    for out in outputs:
        # Variables can't be control dependencies.
        if not isinstance(out, variables.Variable):
            return [out]  # Return first Tensor or Op from outputs.
    return []  # No viable Tensor or Op to use for control deps.
