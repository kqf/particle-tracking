
# NB: Use this sort of tests to check the dataset


def test_data(first_event):
    event_id, hits, cells, particles, truth = first_event
    print("Hits")
    print(hits.head())

    print("Cells")
    print(cells.head())

    print("Particles")
    print(particles.head())

    print("Truth")
    print(truth.head())
