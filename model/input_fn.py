"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def _parse_function(image, label, size):
    """Reshape the image and convert to float value (for both training and validation).

    The following operations are applied:
        - Reshape to [None, size, size, 1]
        - Convert to float
    """
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.reshape(image, [size, size, 1])

    return resized_image, label


def input_fn(is_training, images, labels, params):
    """Input function for the FASHION-MNIST dataset.

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        images: (np.array) images as a big numpy array
        labels: (np.array) corresponding labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(images)
    assert len(images) == len(labels), "Images and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda img, l: _parse_function(img, l, params.image_size)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .repeat(params.num_epochs)  # repeat for multiple epochs
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Return the dataset for tf.estimator (TensorFlow version >= 1.6)
    return dataset
