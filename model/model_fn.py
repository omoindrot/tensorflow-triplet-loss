"""Define the model."""

import numpy as np
import tensorflow as tf


def build_model(is_training, images, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.get_shape().as_list() == [None, 7, 7, num_channels * 2]

    out = tf.reshape(out, [-1, 7 * 7 * num_channels * 2])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is 1.0 iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin):
    """Builds the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Triplet loss: || a - p ||^2 - || a - n ||^2 + margin
    #             = ||p||^2 - 2 <a, p> - ||n||^2 + 2 <a, n> + margin
    # TODO: constrain embeddings to have norm 1?

    # Get dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings, [1, 0]))

    # Get squared L2 norm for each embedding
    square_norm = tf.reduce_sum(tf.square(embeddings), axis=1)  # shape (batch_size,)

    # shape (1, batch_size, 1)
    positive_norm = tf.expand_dims(tf.expand_dims(square_norm, axis=0), axis=2)

    # shape (1, 1, batch_size)
    negative_norm = tf.expand_dims(tf.expand_dims(square_norm, axis=0), axis=0)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dot_product = tf.expand_dims(dot_product, axis=2)

    # shape (batch_size, 1, batch_size)
    anchor_negative_dot_product = tf.expand_dims(dot_product, axis=1)

    triplet_loss = positive_norm - 2 * anchor_positive_dot_product - \
                   negative_norm + 2 * anchor_negative_dot_product + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = mask * triplet_loss

    # Remove negative losses
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 0.0), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / num_valid_triplets

    # Get final mean triplet loss over the positive valid triplets
    # TODO: if num_positive_triplets == 0, return 0 loss
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-10)

    return triplet_loss, fraction_positive_triplets


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')

    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 1], images

    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        embeddings = build_model(is_training, images, params)
        #predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin)
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            #'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('fraction_positive_triplets', fraction)
    #tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', images)

    """
    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(images, mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)
    """

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    #model_spec['predictions'] = predictions
    model_spec['embeddings'] = embeddings
    model_spec['loss'] = loss
    #model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
