import unittest
import numpy as np

from trackml.dataset import load_dataset
from trackml.score import score_event
from model.dbscan import dbscan_model
from model.utils import create_one_event_submission


TRAIN_DATASET = "data/train_100_events"
TEST_DATASET = "data/test/"


class TestDBSCAN(unittest.TestCase):

    def test_training_set(self):
        dataset_submissions = []
        dataset_scores = []

        end_id = 3
        model = dbscan_model(eps=0.01)
        for event_id, hits, cells, particles, truth in load_dataset(
                TRAIN_DATASET):

            if event_id >= end_id:
                break

            # Track pattern recognition
            labels = model.fit_predict(hits)
            print(labels, len(labels))
            print("HERE")

            # Prepare submission for an event
            one_submission = create_one_event_submission(
                event_id, hits, labels)
            dataset_submissions.append(one_submission)

            # Score for the event
            score = score_event(truth, one_submission)
            dataset_scores.append(score)

            print("Score for event %d: %.3f" % (event_id, score))

        print('Mean score: %.3f' % (np.mean(dataset_scores)))
