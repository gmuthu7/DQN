import pytest

from utility.utility import set_random_all


@pytest.fixture
def myseed():
    set_random_all(10)
