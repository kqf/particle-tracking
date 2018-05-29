from trackml.dataset import load_event


# NB: Use this sort of tests to check the dataset
def test_data():
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
