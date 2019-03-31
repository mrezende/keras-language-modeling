import os
import shutil
import datetime

class CompileResults:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def save_results(self):
        base_folder = datetime.datetime.now().strftime(os.path.join('reports', 'results_%m_%d_%Y', '%H_%M_%S'))

        plot_folder = 'plots'
        result_plots_folder = os.path.join(base_folder, plot_folder)
        self.move(plot_folder, result_plots_folder)

        models_folder = 'models'
        result_models_folder = os.path.join(base_folder, models_folder)
        self.move(models_folder, result_models_folder)

        result_conf_file = os.path.join(base_folder, self.conf_file)
        self.move(self.conf_file, result_conf_file)

    def move(self, src, dest):
        shutil.move(src, dest)
