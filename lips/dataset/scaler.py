import numpy as np

class Scaler(object):

    def __init__(self):
        self._m_x = None
        self._m_y = None
        self._std_x = None
        self._std_y = None

    def fit(self, x, y):
        self._m_x = np.mean(x, axis=0)
        self._m_y = np.mean(y, axis=0)
        self._std_x = np.std(x, axis=0)
        self._std_y = np.std(y, axis=0)
        # to avoid division by 0.
        self._std_x[np.abs(self._std_x) <= 1e-1] = 1
        self._std_y[np.abs(self._std_y) <= 1e-1] = 1

    def transform(self, x, y):
        x -= self._m_x
        x /= self._std_x
        y -= self._m_y
        y /= self._std_y

        return x, y

    def fit_transform(self, x, y):
        self._m_x = np.mean(x, axis=0)
        self._m_y = np.mean(y, axis=0)
        self._std_x = np.std(x, axis=0)
        self._std_y = np.std(y, axis=0)

        # to avoid division by 0.
        self._std_x[np.abs(self._std_x) <= 1e-1] = 1
        self._std_y[np.abs(self._std_y) <= 1e-1] = 1

        x -= self._m_x
        x /= self._std_x
        y -= self._m_y
        y /= self._std_y

        return x, y

    def inverse_transform(self, pred_y):
        pred_y *= self._std_y
        pred_y += self._m_y
        return pred_y
