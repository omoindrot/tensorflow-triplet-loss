"""Train the model"""

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

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Create the input data pipeline
    tf.logging.info("Creating the datasets...")
    data = tf.contrib.learn.datasets.mnist.load_mnist(args.data_dir)

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = data.train.num_examples
    params.eval_size = data.test.num_examples

    # Create the two input functions over the two datasets
    train_input_fn = lambda: input_fn(True, data.train.images, data.train.labels, params)
    test_input_fn = lambda: input_fn(False, data.test.images, data.test.labels, params)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(train_input_fn)

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(test_input_fn)
    for key in res:
        print("{}: {}".format(key, res[key]))
