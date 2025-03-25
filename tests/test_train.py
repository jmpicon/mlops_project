from sklearn.linear_model import LogisticRegression
import numpy as np

def test_model_accuracy():
    model = LogisticRegression()
    X = np.random.rand(10, 8)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    assert model.score(X, y) >= 0.5
