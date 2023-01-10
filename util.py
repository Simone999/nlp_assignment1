import functools
import itertools
import os
import random
import re
from typing import Iterable, List
import numpy as np

import tensorflow as tf
import keras as ks
from keras import backend as K
from sklearn.metrics import f1_score


def get_variables(module_name):
    module = globals().get(module_name, None)
    variables = {}
    if module:
        variables = {key: value for key, value in module.__dict__.items() if not (
            key.startswith('__') or key.startswith('_'))}
    return variables


def flatten(seq: Iterable[Iterable]) -> List:
    return [item for inner in seq for item in inner]


def has_digits(input: str) -> bool:
    return any(char.isdigit() for char in input)


def is_number(input: str) -> bool:
    return bool(re.match(r'^[\-]?[0-9]*[\.,]?[0-9]+$', input))


def starts_with_uppercase(input: str) -> bool:
    return str.isupper(input[0])


def remove_items(dictionary: dict, keys: Iterable):
    for key in keys:
        dictionary.pop(key, None)


def generate_combinations(dictionary: dict) -> list[dict]:
    keys, values = zip(*dictionary.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def one_of_as_regex(items: Iterable[str]):
    items = map(re.escape, items)
    items = ''.join(items)
    return fr'[{items}]'

def split_dataset(dataset):
    train_data = dataset[dataset['split'] == 'train']
    val_data = dataset[dataset['split'] == 'validation']
    test_data = dataset[dataset['split'] == 'test']
    return train_data, val_data, test_data


@tf.function
def tf_in(x, y) -> tf.Tensor:
    '''
    Returns True for each element of x that is contained in y
    '''
    # Exploit broadcasting
    # x:      N, 1
    # y:      1, M
    # x==y :  N x M
    # boolean mask:   N

    x = tf.expand_dims(x, axis=-1)  # x:  N x 1
    return tf.math.reduce_any(x == y, axis=-1)


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['SEED'] = str(seed)


def current_seed():
    return int(os.environ['SEED'])

def get_unbatched_labels(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.ragged.stack([y for x, y in dataset.unbatch()]).to_tensor(default_value=0)

@tf.function
def compute_blacklist_mask(y_true: tf.Tensor, black_list: Iterable) -> tf.Tensor:
  return ~tf_in(y_true, black_list)




# class LabelMaskLayer(keras.layers.Masking):
#     def __init__(self, labels: Iterable) -> None:
#         super(LabelMaskLayer, self).__init__()
#         self.labels = labels
    
#     def compute_mask(self, inputs, mask=None):
#         # Also split the mask into 2 if it presents.
#         if mask is None:
#             return None
#         return tf.split(mask, 2, axis=1)
