'''
Model for sequence to sequence learning. The model learns to generate a question given an answer,
and generalizes to other questions and answers.
'''

from __future__ import print_function

import os
import sys

import numpy as np
from keras.engine import Input
from keras.layers import RepeatVector, TimeDistributed, Dense, Activation, merge, ActivityRegularization, GRU, Embedding, \
    regularizers
from keras.models import Model
from keras.optimizers import Adam

try:
    import cPickle as pickle
except:
    import pickle

data_path = os.environ['INSURANCE_QA']
model_save = os.path.join(os.environ['MODEL_PATH'], 'model.h5')


class InsuranceQA:
    def __init__(self):
        self.vocab = self.load('vocabulary')
        self.table = InsuranceQA.VocabularyTable(self.vocab.values())

    def load(self, name):
        return pickle.load(open(os.path.join(data_path, name), 'rb'))

    class VocabularyTable:
        ''' Identical to CharacterTable from Keras example '''

        def __init__(self, words):
            self.words = sorted(set(words))
            self.words_indices = dict((c, i) for i, c in enumerate(self.words))
            self.indices_words = dict((i, c) for i, c in enumerate(self.words))

        def encode(self, sentence, maxlen, one_hot=False):
            if one_hot:
                indices = np.zeros((maxlen, len(self.words)), dtype=np.int32)
                for i, w in enumerate(sentence):
                    if i == maxlen: break
                    indices[i, self.words_indices[w]] = 1
                return indices
            else:
                indices = np.zeros((maxlen,), dtype=np.int32)
                for i, w in enumerate(sentence):
                    if i == maxlen: break
                    indices[i] = self.words_indices[w]
                return indices

        def decode(self, indices, calc_argmax=True):
            if calc_argmax:
                indices = np.argmax(indices, axis=-1)
                # indices = [self.sample(i) for i in indices]
            return ' '.join(self.indices_words[x] for x in indices if x != 0)

        def sample(self, index, noise=0.2):
            index = np.log(index) / noise
            index = np.exp(index) / np.sum(np.exp(index))
            index = np.argmax(np.random.multinomial(1, index, 1))
            return index


def get_model(question_maxlen, answer_maxlen, vocab_len, n_hidden, load_save=False):
    answer = Input(shape=(answer_maxlen,), dtype='int32')
    embedded = Embedding(input_dim=vocab_len, output_dim=n_hidden, mask_zero=True)(answer)
    # masked = Masking(mask_value=0.)(answer)

    # encoder rnn
    encode_rnn = GRU(n_hidden, return_sequences=True, dropout_U=0.2)(embedded)
    encode_rnn = GRU(n_hidden, return_sequences=False, dropout_U=0.2)(encode_rnn)

    encode_brnn = GRU(n_hidden, return_sequences=True, go_backwards=True, dropout_U=0.2)(embedded)
    encode_brnn = GRU(n_hidden, return_sequences=False, go_backwards=True, dropout_U=0.2)(encode_brnn)

    # repeat it maxlen times
    repeat_encoding_rnn = RepeatVector(question_maxlen)(encode_rnn)
    repeat_encoding_brnn = RepeatVector(question_maxlen)(encode_brnn)

    # decoder rnn
    decode_rnn = GRU(n_hidden, return_sequences=True, dropout_U=0.2, dropout_W=0.5)(repeat_encoding_rnn)
    decode_rnn = GRU(n_hidden, return_sequences=True, dropout_U=0.2)(decode_rnn)

    decode_brnn = GRU(n_hidden, return_sequences=True, go_backwards=True, dropout_U=0.2, dropout_W=0.5)(
        repeat_encoding_brnn)
    decode_brnn = GRU(n_hidden, return_sequences=True, go_backwards=True, dropout_U=0.2)(decode_brnn)

    merged_output = merge([decode_rnn, decode_brnn], mode='concat', concat_axis=-1)

    # output
    dense = TimeDistributed(Dense(vocab_len, activity_regularizer=regularizers.activity_l1(1e-4)))(merged_output)
    softmax = Activation('softmax')(dense)

    # compile the prediction model
    model = Model([answer], [softmax])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    if os.path.exists(model_save) and load_save:
        try:
            model.load_weights(model_save)
        except Exception:
            print('A model exists at "%s", but it is not compatible' % model_save)

    return model


if __name__ == '__main__':
    question_maxlen, answer_maxlen = 20, 60

    qa = InsuranceQA()
    batch_size = 50
    n_test = 5
    nb_epoch = 20
    nb_iteration = 200

    print('Generating data...')
    answers = qa.load('answers')


    def gen_questions(batch_size, test=False):
        if test:
            questions = qa.load('test1')
        else:
            questions = qa.load('train')
        while True:
            i = 0
            question_idx = np.zeros(shape=(batch_size, question_maxlen, len(qa.vocab)))
            answer_idx = np.zeros(shape=(batch_size, answer_maxlen))
            for s in questions:
                if test:
                    ans = s['good']
                else:
                    ans = s['answers']
                for a in ans:
                    answer = qa.table.encode([qa.vocab[x] for x in answers[a]], answer_maxlen, one_hot=False)
                    question = qa.table.encode([qa.vocab[x] for x in s['question']], question_maxlen, one_hot=True)
                    # question = np.amax(question, axis=0, keepdims=False)
                    answer_idx[i] = answer
                    question_idx[i] = question
                    i += 1
                    if i == batch_size:
                        yield ([answer_idx], [question_idx])
                        i = 0


    gen = gen_questions(batch_size)
    test_gen = gen_questions(n_test, test=True)

    print('Generating model...')
    model = get_model(question_maxlen=question_maxlen, answer_maxlen=answer_maxlen, vocab_len=len(qa.vocab),
                      n_hidden=256, load_save=False)

    print('Training model...')
    for iteration in range(1, nb_iteration + 1):
        print('\n' + '-' * 50 + '\nIteration %d' % iteration)
        model.fit_generator(gen, samples_per_epoch=100 * batch_size, nb_epoch=nb_epoch)
        model.save_weights(model_save, overwrite=True)

        # test this iteration on some sample data
        x, y = next(test_gen)
        pred = model.predict(x, verbose=0)
        y = y[0]
        x = x[0]
        for i in range(n_test):
            print('Answer: {}'.format(qa.table.decode(x[i], calc_argmax=False)))
            print('  Expected: {}'.format(qa.table.decode(y[i])))
            print('  Predicted: {}'.format(qa.table.decode(pred[i])))