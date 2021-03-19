"""
Copyright 2021 Arthur Müller and Vishal Rangras
This file is part of LemgoRL.

LemgoRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LemgoRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LemgoRL.  If not, see <http://www.gnu.org/licenses/>.

----------------
Utility class containing various methods to plot graphs
and the LongestQueueFirst policy for one-intersection environment.
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


class CustomPlot:

    @staticmethod
    def save_combined_plot(path, xlabel, ylabel, plot_label, t, y_data_list, algo_list):
        fig = plt.figure()
        ax = fig.gca()

        for y, algo in zip(y_data_list, algo_list):
            ax.plot(t, y, label=algo)

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plot_label)
        fig.savefig(path, dpi=300)

    @staticmethod
    def save_combined_ci_plot(path, xlabel, ylabel, plot_label, t, y_data_list, algo_list, std_err_list, alpha=0.15):
        fig = plt.figure()
        ax = fig.gca()

        for ydata, std_err, algo in zip(y_data_list, std_err_list, algo_list):
            y_lb = ydata-std_err
            y_ub = ydata+std_err
            y_lb[y_lb < 0] = 0.0
            ax.fill_between(t, y_ub, y_lb, alpha=alpha)
            ax.plot(t, ydata, label=algo)

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plot_label)
        fig.savefig(path, dpi=300)

    @staticmethod
    def save_plot(path, xlabel, ylabel, plot_label, t, y_data, algo):
        plt.plot(t, y_data, label=algo)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_label)
        plt.savefig(path, dpi=300)
        plt.clf()

    @staticmethod
    def save_ci_plot(path, xlabel, ylabel, plot_label, t, ydata, lb, ub, algo, alpha=0.15):
        lb[lb < 0] = 0.0
        plt.fill_between(t, ub, lb, alpha=alpha)
        plt.plot(t, ydata, label=algo)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_label)
        plt.savefig(path, dpi=300)
        plt.clf()

    @staticmethod
    def plot_figure():
        plt.figure()
