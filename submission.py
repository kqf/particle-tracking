import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


class PrepareInput(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, hits):
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x / r
        hits['y2'] = y / r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z / r
        return hits[['x2', 'y2', 'z2', 'volume_id']].values


def dbscan_model(eps):
    clustering = DBSCAN(eps=eps,
                        metric='minkowski',
                        metric_params={'p': 5},
                        min_samples=1, algorithm='kd_tree')

    model = make_pipeline(
        PrepareInput(),
        StandardScaler(),
        clustering
    )
    return model

# Submission creator


import numpy as np
import pandas as pd


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(
        ([event_id] * len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=[
                              "event_id", "hit_id", "track_id"]).astype(int)
    return submission


# The dataset

import unittest
import numpy as np
import pandas as pd

from trackml.dataset import load_dataset
from trackml.score import score_event
from model.dbscan import dbscan_model
from model.utils import create_one_event_submission

TRAIN_DATASET = "data/train_1/"
TEST_DATASET = "data/test/"


class TestDBSCAN(unittest.TestCase):

    def test_training_set(self):
        dataset_submissions = []
        dataset_scores = []

        # start_id = 1000
        # end_id = start_id + 10

        for event_id, hits, cells, particles, truth in load_dataset(
                TRAIN_DATASET):

            # if event_id >= end_id:
            #     break

            # Track pattern recognition
            model = dbscan_model(eps=0.0008)
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

    def test_test_set(self):
        test_dataset_submissions = []

        for event_id, hits, cells in load_dataset(
                TEST_DATASET, parts=['hits', 'cells']):

            # Track pattern recognition
            model = dbscan_model(eps=0.008)
            labels = model.fit_predict(hits)
            print(labels, len(labels))

            # Prepare submission for an event
            one_submission = create_one_event_submission(
                event_id, hits, labels)
            test_dataset_submissions.append(one_submission)

            print('Event ID: ', event_id)

        # Create submission file
        submussion = pd.concat(test_dataset_submissions, axis=0)
        submussion.to_csv('submission.csv.gz', index=False, compression='gzip')


def main():
    print("Start the analysis")
    suite = TestDBSCAN()
    suite.test_test_set()
    print("Stop the analysis")


if __name__ == '__main__':
    main()
