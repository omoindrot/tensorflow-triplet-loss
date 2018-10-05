"""Create the input data pipeline using `tf.data`"""

# import numpy as np
import tensorflow as tf
from tensorflow.contrib.data.python.ops.interleave_ops import DirectedInterleaveDataset

import model.mnist_dataset as mnist_dataset


def train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.train(data_dir)
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(None)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.test(data_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(None)  # make sure you always have one batch ready to serve
    return dataset


def _make_balanced_batched_dataset(datasets, num_classes, num_classes_per_batch,
                                   num_images_per_class):
    """Create a dataset with balanced batches sampling from multiple datasets.

    For instance if we have 3 datasets representing classes 0, 1 and 2, and we want to create
    batches containing 2 different classes with 3 images each, the labels of a batch could be:
        2, 2, 2, 0, 0, 0
    Or:
        1, 1, 1, 2, 2, 2
    The total batch size in this case is 6.

    Args:
        datasets: (list of Datasets) the datasets to sample from
        num_classes: (int) number of classes, each dataset represents one class
        num_classes_per_batch: (int) number of different classes composing a batch
        num_images_per_class: (int) number of different images from a class in a batch
    """
    assert len(datasets) == num_classes,\
           "There should be {} datasets, got {}".format(num_classes, len(datasets))

    # def generator():
    #     while True:
    #         # Sample the labels that will compose the batch
    #         labels = np.random.choice(range(num_classes),
    #                                   num_classes_per_batch,
    #                                   replace=False)
    #         for label in labels:
    #             for _ in range(num_images_per_class):
    #                 yield label

    # selector = tf.data.Dataset.from_generator(generator, tf.int64)

    def generator(_):
        # Sample `num_classes_per_batch` classes for the batch
        sampled = tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
        # Repeat each element `num_images_per_class` times
        batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_images_per_class])
        return tf.to_int64(tf.reshape(batch_labels, [-1]))

    selector = tf.contrib.data.Counter().map(generator)
    selector = selector.apply(tf.contrib.data.unbatch())

    dataset = DirectedInterleaveDataset(selector, datasets)

    # Batch
    batch_size = num_classes_per_batch * num_images_per_class
    dataset = dataset.batch(batch_size)

    return dataset


def balanced_train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset with balanced batches.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    mnist = mnist_dataset.test(data_dir)

    # pylint: disable=cell-var-from-loop
    datasets = [mnist.filter(lambda img, lab: tf.equal(lab, i)) for i in range(params.num_labels)]

    dataset = _make_balanced_batched_dataset(datasets,
                                             params.num_labels,
                                             params.num_classes_per_batch,
                                             params.num_images_per_class)

    # TODO: check that `buffer_size=None` works
    dataset = dataset.prefetch(None)

    return dataset


def balanced_test_input_fn(data_dir, params):
    """Test input function for the MNIST dataset with balanced batches.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    mnist = mnist_dataset.test(data_dir)

    # pylint: disable=cell-var-from-loop
    datasets = [mnist.filter(lambda img, lab: tf.equal(lab, i)) for i in range(params.num_labels)]

    dataset = _make_balanced_batched_dataset(datasets,
                                             params.num_labels,
                                             params.num_classes_per_batch,
                                             params.num_images_per_class)

    # TODO: check that `buffer_size=None` works
    dataset = dataset.prefetch(None)

    return dataset
