import matplotlib
# matplotlib.use('Agg')
import numpy as np
import os
import matplotlib.pyplot as plt

import pandas as pd


class ReportResult:
    def __init__(self, data, index, plot_name):
        self.df = df = pd.DataFrame(data, index=index)
        self.plot_name = plot_name

    def generate_line_report(self):
        plot = self.df.plot.line()
        self.fig = plt.gcf()
        plt.show()
        return plot


    def save_plot(self):
        if not os.path.exists('plots/'):
            os.makedirs('plots/')
        filename = f'{self.plot_name}_plot.png'
        self.fig.savefig(f'plots/{filename}')
        return filename
