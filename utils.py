"""
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
