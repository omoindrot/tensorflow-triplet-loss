"""Test a new data pipeline with balanced batches.

To train a network with triplet loss we need to build batches in a special way.
Each batch should contain for instance 5 different classes with 10 images in each class,
for a total of 50 images.

This allows to have useful triplets to train on. Otherwise, if the total number of classes is high,
it is possible to have a batch with only different classes and therefore no triplet is valid.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data.python.ops.interleave_ops import DirectedInterleaveDataset

import model.mnist_dataset as mnist_dataset
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the data pipeline
    # TODO: put this new pipeline in `model/input_fn.py`
    mnist = mnist_dataset.train(args.data_dir)

    # pylint: disable=cell-var-from-loop
    datasets = [mnist.filter(lambda img, lab: tf.equal(lab, i)) for i in range(params.num_labels)]

    # TODO: put these in params
    num_classes_per_batch = 5
    num_images_per_class = 10

    def generator():
        while True:
            # Sample the labels that will compose the batch
            labels = np.random.choice(range(params.num_labels),
                                      num_classes_per_batch,
                                      replace=False)
            for label in labels:
                for _ in range(num_images_per_class):
                    yield label

    selector = tf.data.Dataset.from_generator(generator, tf.int64)
    dataset = DirectedInterleaveDataset(selector, datasets)

    batch_size = num_classes_per_batch * num_images_per_class
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    x = dataset.make_one_shot_iterator().get_next()

    sess = tf.Session()

    res = sess.run(x)
    print(res[1])
