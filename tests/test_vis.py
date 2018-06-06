from mpl_toolkits.mplot3d import Axes3D  # noqa, enable 3D projection
import matplotlib.pyplot as plt


def plot_true(hits):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hits.tx, hits.ty, hits.tz)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def test_training_set(training_set):
    # Event loop
    for event_id, hits, cells, particles, truth in training_set:
        print(truth.particle_id.count())
        print(len(truth))
        plot_true(truth)
