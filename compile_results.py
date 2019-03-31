import os
import shutil
import datetime


class CompileResults:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def save_training_results(self):
        base_folder = datetime.datetime.now().strftime(
            os.path.join('reports', 'training', 'results_%m_%d_%Y', '%H_%M_%S'))

        plot_folder = 'plots'
        archive_plots_folder = os.path.join(base_folder, plot_folder)
        self.move(plot_folder, archive_plots_folder)

        models_folder = 'models'
        archive_models_folder = os.path.join(base_folder, models_folder)
        self.move(models_folder, archive_models_folder)

        archive_conf_file = os.path.join(base_folder, self.conf_file)
        self.move(self.conf_file, archive_conf_file)

        conf_list_file = 'conf_list.txt'
        archive_conf_names = os.path.join(base_folder, conf_list_file)
        self.move(conf_list_file, archive_conf_names)

    def save_predict_results(self):
        base_folder = datetime.datetime.now().strftime(
            os.path.join('reports', 'prediction', 'results_%m_%d_%Y', '%H_%M_%S'))

        models_folder = 'models'
        archive_models_folder = os.path.join(base_folder, models_folder)
        self.move(models_folder, archive_models_folder)

        archive_conf_file = os.path.join(base_folder, self.conf_file)
        self.move(self.conf_file, archive_conf_file)

        score_file = 'results_conf.txt'
        archive_score_file = os.path.join(base_folder, score_file)
        self.move(self.conf_file, archive_score_file)


    def move(self, src, dest):
        shutil.move(os.path.abspath(src), dest)
