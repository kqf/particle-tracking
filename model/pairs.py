import numpy as np
from sklearn.linear_model import SGDClassifier


def build_model():
    return SGDClassifier()


def combinations(x, n=2):
    return np.array(np.meshgrid((x,) * 2)).T.reshape(-1, n)


def make_classification(dataset, imballance=4):
    events = []
    for event_id, hits, cells, particles, truth in dataset:
        hit_cells = cells.groupby(['hit_id']).value.count().values
        hit_value = cells.groupby(['hit_id']).value.sum().values
        features = np.hstack((
            hits[['x', 'y', 'z']] / 1000,
            hit_cells.reshape(len(hit_cells), 1) / 10,
            hit_value.reshape(len(hit_cells), 1))
        )
        particle_ids = truth.particle_id.unique()
        particle_ids = particle_ids[np.where(particle_ids > 0)[0]]
        pair = []
        for pid in particle_ids:
            hit_idx = truth[truth["particle_id"] == pid].hit_id.values - 1
            pair.append(combinations(hit_idx))
        pair = np.vstack(pair)
        signal = np.hstack([features[pair[:, 0]], features[pair[:, 1]],
                            np.ones((pair.shape[0], 1))])

        i = np.random.randint(hits.shape[0], size=signal.shape[0] * imballance)
        j = np.random.randint(hits.shape[0], size=signal.shape[0] * imballance)

        pair = np.hstack((i.reshape(-1, 1), j.reshape(-1, 1)))
        pid = truth.particle_id.values
        pair = pair[((pid[i] == 0) | (pid[i] != pid[j]))]

        background = np.hstack([features[pair[:, 0]], features[pair[:, 1]],
                                np.zeros((pair.shape[0], 1))])

        events.append(np.vstack((signal, background)))
    dataset = np.vstack(events)
    return dataset[:, :-1], dataset[:, -1]
