def test_agent_callback(agent_mock):
    trainer = Trainer(3, 100, agent_mock, agent_mock, 14, agent_mock)
    trainer._agent_callback(100)({"a": 1, "b": 2})
    agent_mock.log_metrics.assert_called_with({"a": 1, "b": 2}, step=100)


@pytest.mark.usefixtures("seed")
def test_evaluate_callback(agent_mock):
    evaluator = Evaluator()
    trainer = Trainer(3, 100, agent_mock, evaluator, 14)
    trainer_callback = TrainerCallback(agent_mock, 10000)
    best_reward = [10]
    fn = trainer_callback.after_evaluate(10, agent_mock, best_reward)
    fn({"eval_ep_rew": np.full((10,), 100), "eval_ep_len": np.full((10,), 100)})
    assert best_reward[0] == 100
