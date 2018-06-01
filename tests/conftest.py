import pytest
import itertools
from trackml.dataset import load_dataset

TRAIN_DATASET = 'data/train_100_events'


@pytest.fixture()
def training_set(n_events=2):
    return itertools.islice(load_dataset(TRAIN_DATASET), n_events)


@pytest.fixture()
def first_event():
    return next(load_dataset(TRAIN_DATASET))
