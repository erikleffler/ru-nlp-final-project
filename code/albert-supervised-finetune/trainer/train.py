import tensorflow as tf

# There is alot of unused code in this file. The only function that is used
# currently is mlm_sparse_categorical_crossentropy. The rest of the functions
# were written when I was trying out gradient accumulation. I left them here
# just in case they turn out to be needed.

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)


@tf.function
def dist_mlm_sparse_categorical_crossentropy(
    labels, predictions, global_batch_size, loss_object
):

    print("trace dist loss")

    mask = tf.not_equal(labels, -1)

    labels_masked = tf.boolean_mask(labels, mask)
    predictions_masked = tf.boolean_mask(predictions, mask)

    per_example_loss = loss_object(labels_masked, predictions_masked)
    return tf.nn.compute_average_loss(
        per_example_loss, global_batch_size=global_batch_size
    )


@tf.function
def mlm_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Calculates the
    """
    print("trace loss")
    mask = tf.not_equal(y_true, -1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true_masked, y_pred_masked, from_logits=True
    )


@tf.function
def dist_train_step_accumulate_gradients(inputs, iter_batch_size, global_batch_size):

    total_loss = 0

    features, labels = inputs

    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]

    dist_batch_size = input_ids.shape[0]

    accumulation_steps = dist_batch_size // iter_batch_size

    train_vars = bert.trainable_variables
    accumulated_gradients = [tf.zeros_like(train_var) for train_var in train_vars]

    for i in range(accumulation_steps):

        input_ids_sub_batch = tf.slice(
            input_ids, [i * iter_batch_size, 0], [iter_batch_size, -1]
        )

        attention_mask_sub_batch = tf.slice(
            attention_mask, [i * iter_batch_size, 0], [iter_batch_size, -1]
        )

        labels_sub_batch = tf.slice(
            labels, [i * iter_batch_size, 0], [iter_batch_size, -1]
        )

        with tf.GradientTape() as tape:
            predictions = bert(
                [input_ids_sub_batch, attention_mask_sub_batch], training=True
            )[0]
            loss = mlm_sparse_categorical_crossentropy(
                labels=labels_sub_batch,
                predictions=predictions,
                global_batch_size=global_batch_size,
            )

        total_loss += loss

        # get gradients of this tape
        gradients = tape.gradient(loss, train_vars)

        # Accumulate the gradients
        accumulated_gradients = [
            accumulated_gradient + gradient
            for accumulated_gradient, gradient in zip(accumulated_gradients, gradients)
        ]

    optimizer.apply_gradients(zip(accumulated_gradients, bert.trainable_variables))

    return total_loss


@tf.function
def train_step(inputs, batch_size):

    features, labels = inputs

    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]

    with tf.GradientTape() as tape:
        predictions = bert(features, training=True)[0]
        loss = mlm_sparse_categorical_crossentropy(
            labels=labels, predictions=predictions, global_batch_size=batch_size
        )

    # get gradients of this tape
    gradients = tape.gradient(loss, bert.trainable_variables)

    optimizer.apply_gradients(zip(gradients, bert.trainable_variables))

    return loss


@tf.function
def dist_train_step(dist_inputs, batch_size):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs, batch_size))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def test_step(inputs, batch_size):

    features, labels = inputs

    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]

    predictions = bert(features, training=False)[0]
    loss = mlm_sparse_categorical_crossentropy(
        labels=labels, predictions=predictions, global_batch_size=batch_size
    )

    return loss


@tf.function
def dist_test_step(dist_inputs, batch_size):
    per_replica_losses = strategy.run(test_step, args=(dist_inputs, batch_size))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
