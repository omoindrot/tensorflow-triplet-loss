"""Test a new data pipeline with balanced batches.

To train a network with triplet loss we need to build batches in a special way.
Each batch should contain for instance 5 different classes with 10 images in each class,
for a total of 50 images.

This allows to have useful triplets to train on. Otherwise, if the total number of classes is high,
it is possible to have a batch with only different classes and therefore no triplet is valid.
"""

import argparse
import os

import tensorflow as tf

from model.input_fn import balanced_train_input_fn
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

    # Create a balanced test dataset
    dataset = balanced_train_input_fn(args.data_dir, params)

    x = dataset.make_one_shot_iterator().get_next()

    sess = tf.Session()

    res = sess.run(x)
    print(res[1])
