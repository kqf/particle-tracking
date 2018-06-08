from model.pairs import make_classification, build_model
from sklearn.model_selection import train_test_split


# 1000 (1077616, 11)
# 1001 (801978, 11)
def test_makes_features(training_set):
    X, y = make_classification(training_set)
    print(X.shape)

    model = build_model()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    model.fit(X_tr, y_tr)
    print("Train", model.score(X_tr, y_tr))
    print("Test ", model.score(X_te, y_te))
