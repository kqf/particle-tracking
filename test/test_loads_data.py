import unittest
from trackml.dataset import load_event


class TestInputData(unittest.TestCase):

    def test_data(self):
        hits, cells, particles, truth = load_event(
            "data/train_100_events/event000001064"
        )
        print('Hits')
        print(hits.head())

        print('Cells')
        print(cells.head())

        print('Particles')
        print(particles.head())

        print('Truth')
        print(truth.head())

    def test_test_sample(self):
        hits, cells = load_event(
            "data/test/event000000070", parts=['hits', 'cells']
        )
        print('Hits')
        print(hits.head())

        print('Cells')
        print(cells.head())