import math
import pytest

from data.data_loader import DataLoader


def assert_decays(quadratic_decay, linear_decay, allowed_error=0.8):
    assert math.fabs(linear_decay - 2) < allowed_error, f'linear decay rate should be ~2 but was {linear_decay}'
    assert quadratic_decay > 4 - allowed_error, f'linear decay rate should be ~4 but was {quadratic_decay}'


class GradientTest:
    @pytest.fixture()
    def data(self):
        data = DataLoader.load_dataset("PeaksData.mat")
        return data

    @pytest.fixture()
    def X(self, data):
        return data['Yt']

    @pytest.fixture()
    def c(self, data):
        return data['Ct']

