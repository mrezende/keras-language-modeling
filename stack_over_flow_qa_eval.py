
from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime, time
from report_result import ReportResult
from configuration import Conf
from archive_results import ArchiveResults

import argparse
import shutil

import pickle
import json
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

import threading
from scipy.stats import rankdata
import logging
import numpy as np
import tensorflow as tf

def clear_session():
    K.clear_session()

def remove_plots():
    shutil.rmtree('plots')

class Evaluator:
    def __init__(self, conf_json, model, optimizer=None):
        try:
            data_path = os.environ['STACK_OVER_FLOW_QA']
        except KeyError:
            print("STACK_OVER_FLOW_QA is not set. Set it to your clone of https://github.com/mrezende/stack_over_flow_python")
            sys.exit(1)
        self.conf = Conf(conf_json)
        self.model = model(self.conf)
        self.name = self.conf.name()

        self.path = data_path
        self.params = self.conf.training_params()
        optimizer = self.params['optimizer'] if optimizer is None else optimizer
        self.model.compile(optimizer)

        self.answers = self.load('answers.json') # self.load('generated')
        self.data = self.load('sof_python_data_labeled.json')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None
        self.top1_ls = []
        self.mrr_ls = []

    ##### Resources #####

    def save_conf(self):
        self.conf.save_conf()

    def load(self, name):
        return json.load(open(os.path.join(self.path, name), 'r'))

    def vocab(self):
        if self._vocab is None:
            reverse_vocab = self.reverse_vocab()
            self._vocab = dict((v, k.lower()) for k, v in reverse_vocab.items())
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            samples = self.load('samples_for_tokenizer.json')

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(samples)

            self._reverse_vocab = tokenizer.word_index
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        logger.info(f'Saving weights: models/weights_epoch_{self.name}.h5')
        self.model.save_weights(f'models/weights_epoch_{self.name}.h5', overwrite=True)

    def load_epoch(self):
        assert os.path.exists(f'models/weights_epoch_{self.name}.h5'), f'Weights at epoch {self.name} not found'
        logger.info(f'Loading weights: models/weights_epoch_{self.name}.h5')
        self.model.load_weights(f'models/weights_epoch_{self.name}.h5')

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.question_len())

    def pada(self, data):
        return self.pad(data, self.conf.answer_len())

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def get_time(self):
        return strftime('%Y-%m-%d %H:%M:%S', gmtime())

    def train_and_evaluate(self, mode='train'):


        X = np.array(self.data)

        val_losses = []
        top1s = []
        mrrs = []

        test_size = self.params['test_size']

        X_train, X_test = train_test_split(
        X, test_size = test_size, random_state = 42)

        if mode == 'train':
            val_loss = self.train(X_train)
            val_losses.append(val_loss)
            logger.info(f'Val loss: {val_loss}')

        elif mode == 'evaluate':
            tf.reset_default_graph()
            self.load_epoch()
            top1, mrr = self.get_score(X_test)
            top1s.append(top1)
            mrrs.append(mrr)
            logger.info(f'Top-1 Precision: {top1}, MRR: {mrr}')



    def train(self, X):
        batch_size = self.params['batch_size']
        validation_split = self.params['validation_split']
        nb_epoch = self.params['nb_epoch']


        # top_50 = self.load('top_50')

        questions = list()
        good_answers = list()

        for j, q in enumerate(X):
            questions += [q['question']] * len(q['good_answers'])
            good_answers += q['good_answers']

        logger.info('Began training at %s on %d samples' % (self.get_time(), len(questions)))

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)


        # According to NN Design Book:
        # For this reason it is best to try several different initial guesses in order to ensure that
        # a global minimum has been obtained.


        best_val_loss = {'loss': 1., 'epoch': 0}
        hist_losses = {'val_loss': [], 'loss': []}

        for i in range(1, nb_epoch + 1):

            bad_answers = self.pada(random.sample(self.answers, len(good_answers)))

            logger.info(f'Fitting epoch {i}')
            hist = self.model.fit([questions, good_answers, bad_answers], epochs=1, batch_size=batch_size,
                                  validation_split=validation_split, verbose=2)

            val_loss = hist.history['val_loss'][0]
            loss = hist.history['loss'][0]
            hist_losses['val_loss'].append(val_loss)
            hist_losses['loss'].append(loss)

            if val_loss < best_val_loss['loss']:
                print(hist.history)
                best_val_loss = {'loss': val_loss, 'epoch': i}

                logger.info('%s -- Epoch %d ' % (self.get_time(), i) +
                            'Loss = %.4f, Validation Loss = %.4f ' % (loss, val_loss) +
                            '(Best: Loss = %.4f, Epoch = %d)' % (best_val_loss['loss'], best_val_loss['epoch']))

                # saving weights
                self.save_epoch()

        # save plot val_loss, loss
        report = ReportResult(hist_losses, [i in range(1, nb_epoch + 1)], self.name)
        plot = report.generate_line_report()
        report.save_plot()

        logger.info(f'saving loss, val_loss plot')


        # save conf
        self.save_conf()

        clear_session()
        return val_loss

    def get_score(self, X):
        c_1, c_2 = 0, 0

        for i, d in enumerate(X):
            answers = d['good_answers'] + d['bad_answers']
            answers = self.pada(answers)
            question = self.padq([d['question']] * len(answers))

            sims = self.model.predict([question, answers])

            n_good = 1
            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])

            r = rankdata(sims, method='max')

            #if verbose:
            #    min_r = np.argmin(sims)
            #    amin_r = self.answers[indices[min_r]]
            #     amax_r = self.answers[indices[max_r]]
            #     amax_n = self.answers[indices[max_n]]
            #
            #     print(' '.join(self.revert(d['question'])))
            #     print('Predicted: ({}) '.format(sims[max_r]) + ' '.join(self.revert(amax_r)))
            #     print('Expected: ({}) Rank = {} '.format(sims[max_n], r[max_n]) + ' '.join(self.revert(amax_n)))
            #     print('Worst: ({})'.format(sims[min_r]) + ' '.join(self.revert(amin_r)))

            c_1 += 1 if max_r == max_n else 0
            c_2 += 1 / float(r[max_r] - r[max_n] + 1)

        top1 = c_1 / float(len(data))
        mrr = c_2 / float(len(data))


        print('Top-1 Precision: %f' % top1)
        print('MRR: %f' % mrr)

        return top1, mrr

    def save_score(self):
        with open('results_conf.txt', 'a+') as append_file:
            conf_json, name = self.conf.conf_json_and_name()
            top1_precisions = ','.join(self.top1_ls)
            mrrs = ','.join(self.mrr_ls)
            append_file.write(f'{name}; {conf_json}; top-1 precision: {top1_precisions}; MRR: {mrrs}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run question answer selection')
    parser.add_argument('--conf_file', metavar='CONF_FILE', type=str, default="stack_over_flow_conf.json", help='conf json file: stack_over_flow_conf.json')
    parser.add_argument('--mode', metavar='MODE', type=str, default="train", help='mode: train|evaluate')

    args = parser.parse_args()

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    conf_file = args.conf_file
    mode = args.mode

    confs = json.load(open(conf_file, 'r'))
    from keras_models import EmbeddingModel, ConvolutionModel, ConvolutionalLSTM



    for conf in confs:
        logger.info(f'Conf.json: {conf}')
        evaluator = Evaluator(conf, model=ConvolutionalLSTM)
        # train and evaluate the model
        evaluator.train_and_evaluate(mode)











