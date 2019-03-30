import json
import hashlib

class Conf:
    def __init__(self, conf):
        if isinstance(conf, str):
            conf = json.load(open(conf, 'rb'))
        self.conf = conf


    def name(self):
        str = json.dumps(conf[0])
        m.update(str.encode('utf-8'))
        return m.hexdigest()[:6]

    def training_params(self):
        return self.conf['training']

    def question_len(self):
        return self.conf.get('question_len', None)

    def answer_len(self):
        return self.conf.get('answer_len', None)

    def similarity_params(self):
        return self.conf.get('similarity', dict())

    def margin(self):
        return self.conf['margin']

    def initial_embed_weights(self):
        return self.conf['initial_embed_weights']

    def n_words(self):
        return self.conf['n_words']