"""Evaluate the model"""

import argparse
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Create the input data pipeline
    tf.logging.info("Creating the datasets...")
    data = tf.contrib.learn.datasets.mnist.load_mnist(args.data_dir)

    # Specify the sizes of the dataset we evaluate on
    params.eval_size = data.test.num_examples

    # Create the test input function
    test_input_fn = lambda: input_fn(False, data.test.images, data.test.labels, params)

    # Define the model
    tf.logging.info("Creating the model...")
    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=args.model_dir)

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on the test set.")
    res = estimator.evaluate(test_input_fn)
    for key in res:
        print("{}: {}".format(key, res[key]))
