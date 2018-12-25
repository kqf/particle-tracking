import numpy as np

from trackml.score import score_event
from model.dbscan import dbscan_model
from model.utils import create_one_event_submission


def test_training_set(training_set):
    dataset_scores = []

    model = dbscan_model(eps=0.01)
    print()
    for event_id, hits, cells, particles, truth in training_set:
        # Track pattern recognition
        labels = model.fit_predict(hits)

        # Prepare submission for an event
        submission = create_one_event_submission(event_id, hits, labels)

        # Score for the event
        score = score_event(truth, submission)
        dataset_scores.append(score)
        print("Score for event %d: %.3f" % (event_id, score))

    print("Mean score: %.3f" % (np.mean(dataset_scores)))
