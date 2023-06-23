from utils.utility import get_cartpole_parameters, construct_parameter_obj


def test_evaluate_cartpole():
    param_dict = get_cartpole_parameters()
    parameters = construct_parameter_obj(param_dict)
