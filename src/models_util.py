"""Small picklable helpers shared by training and prediction."""
import numpy as np


class Col:
    """Wrap a multi-output model so it exposes one output column via predict()."""
    def __init__(self, model, j):
        self.model, self.j = model, j

    def predict(self, X):
        return np.asarray(self.model.predict(X))[:, self.j]
