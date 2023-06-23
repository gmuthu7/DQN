import numpy as np
import pytest

from buffers.experience_replay import ExperienceReplay, Experience


@pytest.fixture
def experience_buffer() -> ExperienceReplay:
    buffer = ExperienceReplay[Experience](10, 6)
    for _ in range(100):
        experience = Experience(tuple([1, 2, 3]), 0.5, tuple([1, 2, 3]), 1.0, True)
        buffer.store(experience)
    return buffer


def test_experience_sample(experience_buffer):
    assert len(experience_buffer.deque) == 10
    bstate, baction, bnext_state, breward, bdone = experience_buffer.sample()
    assert bstate.shape == (6, 3)
    assert type(bstate) == np.ndarray
