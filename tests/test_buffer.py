import numpy as np
import pytest

from buffers.experience_replay import ExperienceReplay


@pytest.fixture
def buffer() -> ExperienceReplay:
    buffer = ExperienceReplay(2, 2)
    return buffer


def test_sample(buffer):
    test_store(buffer)
    a = buffer.sample()
    assert a[0].shape[0] == buffer.batch_size


@pytest.mark.usefixtures("myseed")
def test_store(buffer):
    num_envs = 1
    action, next_state, reward, state, terminated = get_sars(num_envs)
    buffer.store(state, action, next_state, reward, terminated)
    buffer.store(state, action, next_state, reward, terminated)
    state = np.zeros((1, 4))
    buffer.store(state, action, next_state, reward, terminated)
    assert np.array_equal(state.squeeze(), buffer.state[1])


def get_sars(num_envs):
    state = np.ones((num_envs, 4))
    action = np.zeros((num_envs,))
    next_state = np.zeros((num_envs, 4))
    reward = np.full((num_envs,), 1.3)
    terminated = np.full((num_envs,), False)
    return action, next_state, reward, state, terminated
