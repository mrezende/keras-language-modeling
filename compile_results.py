import os
import shutil
import datetime

class CompileResults:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def save_results(self):
        base_folder = datetime.datetime.now().strftime(os.path.join('reports', 'results_%m_%d_%Y', '%H_%M_%S'))
        plot_folder = 'plots'
        plots = os.path.join(base_folder, plot_folder)
        models_folder = 'models'
        models = os.path.join(base_folder, models_folder)

        conf_file = os.path.join(base_folder, self.conf_file)

        shutil.move(plot_folder, )

    def move(self, file):
        shutil.move(file, )
