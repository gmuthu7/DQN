from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class ErrorPlotter:
    def __init__(self, fig_name: str):
        self.metrics_std = np.array([])
        self.metrics_mean = np.array([])
        self.x_axis = np.array([])
        self.fig: Figure = plt.figure(figsize=(10, 5))
        self.fig.suptitle(fig_name)
        self.ax = self.fig.add_subplot()

    def add_point(self, metrics: Iterable, x: int):
        self.metrics_mean = np.append(self.metrics_mean, [np.mean(metrics)])
        self.metrics_std = np.append(self.metrics_std, [np.std(metrics)])
        self.x_axis = np.append(self.x_axis, [x])

    def plt_fig(self):
        self.ax.cla()
        self.ax.plot(self.x_axis, self.metrics_mean, '-', color='gray')
        self.ax.fill_between(self.x_axis, self.metrics_std + self.metrics_mean, self.metrics_mean - self.metrics_std,
                             color='gray', alpha=0.2)
        return self.fig

    def close(self):
        matplotlib.pyplot.close(self.fig)
