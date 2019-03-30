import matplotlib
matplotlib.use('Agg')
import numpy as np

import pandas as pd
from ggplot import *

class ReportResult:
    def __init__(self, data, plot_name):
        self.df = df = pd.DataFrame(data)
        self.plot_name = plot_name

    def generate_line_report(self):
        self.df.insert(0, 'epochs', range(0, len(self.df)))
        self.df = pd.melt(self.df, id_vars=['epochs'])
        plot = ggplot(aes(x='epochs', y='value', color='variable'), data=self.df) + geom_line()
        filename = f'{self.plot_name}.png'
        plot.save(filename)

