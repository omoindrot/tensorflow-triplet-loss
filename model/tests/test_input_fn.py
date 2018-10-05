"""Test for the inputs functions with tf.data.

Especially test how the balanced batches functions work.
"""

import numpy as np
import tensorflow as tf

from model.input_fn import _make_balanced_batched_dataset


def test_make_balanced_batched_dataset():
    """Test the balanced batched dataset function."""
    num_classes = 10
    num_classes_per_batch = 5
    num_images_per_class = 3
    batch_size = num_classes_per_batch * num_images_per_class

    datasets = [tf.data.Dataset.range(i * 100, i * 100 + 100) for i in range(num_classes)]

    dataset = _make_balanced_batched_dataset(datasets, num_classes,
                                             num_classes_per_batch, num_images_per_class)

    x = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        res = sess.run(x)
        # res should be like [300, 301, 302, 500, 501, 502, 0, 1, 2, 800, 801, 802, 900, 901, 902]

        assert len(res) == batch_size,\
               "There should be {} elements, got {}".format(batch_size, len(res))

        quotient_1, remainder_1 = np.divmod(res, 100)

        # quotient is the class of the data, should have 5 different ones
        assert quotient_1[0] == quotient_1[1]
        assert quotient_1[0] == quotient_1[2]
        assert len(np.unique(quotient_1)) == num_classes_per_batch,\
               "There should be only {} different classes, got {}".format(
                       num_classes_per_batch, len(np.unique(quotient_1)))

        # remainder is the number of the data point in the dataset
        correct_remainder_1 = np.tile(np.arange(num_images_per_class), num_classes_per_batch)
        assert np.all(remainder_1 == correct_remainder_1)

        res = sess.run(x)
        # res should be like [200, 201, 202, 3, 4, 5, 700, 701, 702, 400, 401, 402, 303, 304, 305]

        quotient_2, remainder_2 = np.divmod(res, 100)

        # quotient is the class of the data, should have 5 different ones
        assert quotient_2[0] == quotient_2[1]
        assert quotient_2[0] == quotient_2[2]
        assert len(np.unique(quotient_2)) == num_classes_per_batch,\
               "There should be only {} different classes, got {}".format(
                       num_classes_per_batch, len(np.unique(quotient_2)))

        # remainder is the number of the data point in the dataset
        # TODO: check the remainder_2
        print(remainder_2)
