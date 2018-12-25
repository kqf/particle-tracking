from mpl_toolkits.mplot3d import Axes3D  # noqa, enable 3D projection
import matplotlib.pyplot as plt


def plot_true(hits):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(hits.tx, hits.ty, hits.tz)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z coordinate")
    plt.show()


def test_visualize(training_set):
    # Event loop
    for event_id, hits, cells, particles, truth in training_set:
        mc_particles = truth[truth["particle_id"] != 0]
        track_lengths = mc_particles[["particle_id", "weight"]].groupby(
            "particle_id",
            as_index=False).count()
        largest = track_lengths.sort_values("weight")[-4:]["particle_id"]
        to_plot = truth[truth["particle_id"].isin(largest)]
        plot_true(to_plot)
