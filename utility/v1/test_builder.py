import gymnasium
import numpy as np
import pytest
from gymnasium.wrappers import RecordEpisodeStatistics


@pytest.mark.mytest
def test_agent():
    envs = gymnasium.vector.make(id="FrozenLake-v1", num_envs=2, is_slippery=True)
    envs = RecordEpisodeStatistics(envs)
    a = np.array([0, 1])
    envs.reset()
    b = envs.step(a)
    for i in range(100):
        print(envs.step([0, 0]))
