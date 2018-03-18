# Triplet loss in TensorFlow

*Author: Olivier Moindrot*

This code is adapted from code I wrote for [CS230](https://cs230-stanford.github.io) in [this repository](https://github.com/cs230-stanford/cs230-code-examples) at `tensorflow/vision`.


## Requirements

We recommend using python3 and a virtual environment.
The default `venv` should be used, or `virtualenv` with `python3`.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Task


## Download the dataset

```bash
python build_dataset.py --data_dir data
```

## Test

Run all the tests:
```bash
nosetests tests
```

Run a specific test:
```bash
python -m tests.test_triplet_loss
```


## Resources

Introduction to the `tf.data` pipeline
- [programmer's guide](https://www.tensorflow.org/programmers_guide/datasets)
- [reading images](https://www.tensorflow.org/programmers_guide/datasets#decoding_image_data_and_resizing_it)
