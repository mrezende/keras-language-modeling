
from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime, time
from report_result import ReportResult
from configuration import Conf
from compile_results import CompileResults

import argparse

import pickle
import json
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

import threading
from scipy.stats import rankdata
import logging

random.seed(42)

def clear_session():
    K.clear_session()


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

        self.model.save_weights(f'models/weights_epoch_{self.name}.h5', overwrite=True)

    def load_epoch(self):
        assert os.path.exists(f'models/weights_epoch_{self.name}.h5'), f'Weights at epoch {self.name} not found'
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

    def train(self):
        batch_size = self.params['batch_size']
        validation_split = self.params['validation_split']
        epochs = self.params['epochs']

        training_set = self.load('train.json')
        # top_50 = self.load('top_50')

        questions = list()
        good_answers = list()
        indices = list()

        for j, q in enumerate(training_set):
            questions += [q['question']] * len(q['answers'])
            good_answers += [i for i in q['answers']]
            indices += [j] * len(q['answers'])
        logger.info('Began training at %s on %d samples' % (self.get_time(), len(questions)))

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)



        # def get_bad_samples(indices, top_50):
        #     return [self.answers[random.choice(top_50[i])] for i in indices]


        # sample from all answers to get bad answers
        # if i % 2 == 0:
        #     bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))
        # else:
        #     bad_answers = self.pada(get_bad_samples(indices, top_50))
        bad_answers = self.pada(random.sample(self.answers, len(good_answers)))


        hist = self.model.fit([questions, good_answers, bad_answers], epochs=epochs, batch_size=batch_size,
                              validation_split=validation_split, verbose=2)

        # save plot val_loss, loss
        report = ReportResult(hist.history, self.name)
        plot = report.generate_line_report()
        report.save_plot(plot)

        logger.info(f'saving loss, val_loss plot')


        # save_model_architecture(prediction_model, model_name=model_name)
        self.save_epoch()

        # save conf
        self.save_conf()

        clear_session()

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='', file=sys.stderr)
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='', file=sys.stderr)

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['test.json']])
        return self._eval_sets

    def get_score(self, verbose=False):
        top1_ls = []
        mrr_ls = []
        for name, data in self.eval_sets().items():
            print('----- %s -----' % name)

            random.shuffle(data)

            if 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            c_1, c_2 = 0, 0

            for i, d in enumerate(data):
                self.prog_bar(i, len(data))

                answers = d['good'] + d['bad']
                answers = self.pada(answers)
                question = self.padq([d['question']] * len(answers))

                sims = self.model.predict([question, answers])

                n_good = len(d['good'])
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

            del data
            print('Top-1 Precision: %f' % top1)
            print('MRR: %f' % mrr)
            top1_ls.append(top1)
            mrr_ls.append(mrr)
        self.top1_ls = top1_ls
        self.mrr_ls = mrr_ls
        return self.top1_ls, self.mrr_ls

    def save_score(self):
        with open('results_conf.txt', 'a+') as append_file:
            conf_json, name = self.conf.conf_json_and_name()
            top1_precisions = ','.join(self.top1_ls)
            mrrs = ','.join(self.mrr_ls)
            append_file.write(f'{name}; {conf_json}; top-1 precision: {top1_precisions}; MRR: {mrrs}\n')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='run question answer selection')
    parser.add_argument('--mode', metavar='MODE', type=str, default="train", help='mode: train/predict/test')
    parser.add_argument('--conf_file', metavar='MODE', type=str, default="stack_over_flow_conf.json", help='conf json file: stack_over_flow_conf.json')

    args = parser.parse_args()

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    mode = args.mode

    conf_file = args.conf_file

    confs = json.load(open(conf_file, 'r'))
    from keras_models import EmbeddingModel, ConvolutionModel, ConvolutionalLSTM

    if mode == 'train':

        for conf in confs:
            logger.info(f'Conf.json: {conf}')
            evaluator = Evaluator(conf, model=ConvolutionalLSTM)
            # train the model
            evaluator.train()

        # archive models, plots
        compile_results = CompileResults(conf_file)
        compile_results.save_training_results()

    elif mode == 'predict':

        for conf in confs:
            logger.info(f'Conf.json: {conf}')
            evaluator = Evaluator(conf, model=ConvolutionalLSTM)
            # evaluate mrr for a particular epoch
            evaluator.load_epoch()
            top1, mrr = evaluator.get_score(verbose=False)
            logger.info(' - Top-1 Precision:')
            logger.info('   - %.3f on test 1' % top1[0])

            logger.info(' - MRR:')
            logger.info('   - %.3f on test 1' % mrr[0])

            #save score per conf
            evaluator.save_score()

        # archive models, conf file, score results
        compile_results = CompileResults(conf_file)
        compile_results.save_predict_results()





