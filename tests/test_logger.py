import random

import numpy as np
import pytest
from matplotlib import pyplot as plt
from ray import tune

from loggers.error_plotter import ErrorPlotter
from loggers.ray_tune_logger import RayTuneLogger


def test_plot():
    x = tune.loguniform(0.001, 100).sample(size=1000)
    plt.hist(np.log10(x))
    plt.show()
    assert len(x) == 1000


def test_plotter():
    logger = RayTuneLogger("../loggers/logs/mlflowexamples/plot_test")
    plotter = ErrorPlotter(f"mlflowexamples{np.random.rand()}")
    logger.writer.add_scalar("test", 123, 123)
    for i in range(int(1e2)):
        plotter.add_point(np.random.randint(10000) * 1, i)
    fig = plotter.plt_fig()
    logger.log_figure(fig, 100)
    for i in range(int(1e2), int(1e4)):
        plotter.add_point(np.random.randn(10000) * 1000, i)
    fig = plotter.plt_fig()
    logger.log_figure(fig, 200)
