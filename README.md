# Triplet loss in TensorFlow [![Build Status](https://travis-ci.org/omoindrot/tensorflow-triplet-loss.svg?branch=master)](https://travis-ci.org/omoindrot/tensorflow-triplet-loss)
*Author: Olivier Moindrot*

This repository contains a triplet loss implementation in TensorFlow with online triplet mining.
Please check the [blog post][blog] for a full description.

This code is adapted from code I wrote for [CS230](https://cs230.stanford.edu) in [this repository](https://github.com/cs230-stanford/cs230-code-examples) at `tensorflow/vision`.
A set of tutorials for this code can be found [here](https://cs230-stanford.github.io).


## Requirements

We recommend using python3 and a virtual environment.
The default `venv` should be used, or `virtualenv` with `python3`.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements_cpu.txt
```

If you are using a GPU, you will need to install `tensorflow-gpu` so do:
```bash
pip install -r requirements_gpu.txt
```

## Triplet loss

The interesting part, defining triplet loss with triplet mining can be found in [`model/triplet_loss.py`](model/triplet_loss.py).

Everything is explained in the [blog post][blog].

To use the "batch all" version, you can do:
```python
from model.triplet_loss import batch_all_triplet_loss

loss, fraction_positive = batch_all_triplet_loss(labels, embeddings, margin, squared=False)
```

In this case `fraction_positive` is a useful thing to plot in TensorBoard to track the average number of hard and semi-hard triplets.

To use the "batch hard" version, you can do:
```python
from model.triplet_loss import batch_hard_triplet_loss

loss = batch_hard_triplet_loss(labels, embeddings, margin, squared=False)
```

## Training on MNIST

To run a new experiment called "test", do:
```bash
python train.py --model_dir experiments/test
```


## Test

To run all the tests, run this from the project directory:
```bash
pytest
```

To run a specific test:
```bash
python -m model.tests.test_triplet_loss
```


## Resources

- [Blog post][blog] explaining this project.
- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].


[blog]: https://omoindrot.github.io/triplet-loss
[tf-triplet-loss]: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss

