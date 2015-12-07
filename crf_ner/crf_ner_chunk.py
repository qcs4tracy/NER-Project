__author__ = 'Qiu,Chusheng'
import pycrfsuite
import os, shutil, codecs
from collections import Counter
from template import feature_extractor, get_shape, get_type, get_da, boolean

def sent2tokens(sent):
    return [token for token, label, _ in sent]

def sent2labels(sent):
    return [x[-1] for x in sent]

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))


default_train_params = {
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,  # stop earlier
        'feature.possible_transitions': True,  # include transitions that are possible, but not observed
        }

class CrfNeChunkerTaggerBase(object):
    def __init__(self, name='model', train=None, model_path=None):
        """
        Parameters
        ----------
        train, list of (word, pos, ne_tag) tuples
            The train data with structure conforming to nltk's taggers

        name: str
            The file path to a model file

        model_path: str
            The model file path that is to be opened and loaded.
        """

        if not name:
            raise ValueError('the `name` parameter should not be None or empty string.')

        if (train and model_path) or (not train and not model_path):
            raise ValueError('either `train` or `model_path` should specified, but not both.')

        self._model_name = name
        self._model_path = None
        self._data = train
        self._trainer = None
        self._tagger = None
        self._info = None

        # an model file is specified, open the tagger instead of train one
        if model_path:
            self.open(model_path)
            self._model_path = model_path

    def train(self, verbose=False, params=default_train_params, *args, **kwargs):

        # already trained or opened an model, do nothing
        if self._model_path or not self._data:
            return

        # trained before, clear the trainer
        if self._trainer:
            self._trainer.clear()
        else:
            self._trainer = pycrfsuite.Trainer(verbose=verbose)

        # set the parameters
        trainer = self._trainer
        param_names = set(trainer.params())

        if isinstance(params, dict):
            params = {k: v for k, v in params.items() if k in param_names}
            if len(params) > 0:
                trainer.set_params(params)

        kw_params = {name: val for name, val in kwargs.items() if name in param_names}
        if len(kw_params) > 0:
            trainer.set_params(kw_params)

        data = self._data
        # feed the trainer with the data
        for sent in data:
            trainer.append(self._feature_detect(sent), self._seqence_labels(sent))

        model_path = os.path.join(os.getcwd(), self._model_name + r'.crfsuite')
        trainer.train(model_path)
        self._model_path = model_path

    def save(self, path):
        """save the model file to another directory
        Parameters
        ----------
        path: str, the directory or file path
        """
        if os.path.isdir(path):
            shutil.move(self._model_path, path)
        elif path[-1] != os.path.sep:  # path is a file name
            shutil.copyfile(self._model_path, path)
        else:
            raise IOError('%s is not a directory or file path, saving model failed.' % (self._model_name))

    # the ner labels for given sentence (i.e. words sequence)
    def _seqence_labels(self, sent):
        return [x[2] for x in sent]

    def open(self, model_path):
        """
        open the trained model and use it as a tagger
        """
        if self._tagger:
            self._tagger.close()
        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(model_path)

    def _feature_detect(self, sent):
        """
        The method used to extract the features for an sentences consist
         of ('word', 'POS-tag', 'NER-tag') tuples, which must ne implemented by subclasses.
        """
        raise NotImplementedError('_feature_detect() method must be defined.')

    def tag(self, sent):
        """
        Parameters
        ----------
        sent: list of ('word', 'pos', 'ne_tag') or ('word', 'pos') tuples
            The sentence sequence of words.

        :rtype: list of str
        :return: list of NER tags
        """
        if not self._model_path and self._data:
            raise RuntimeError('tagger must be trained first by calling train(), or opened by open() method.')

        if not self._tagger:
            self.open(self._model_path)

        return [codecs.decode(x) for x in self._tagger.tag(self._feature_detect(sent))]

    def model_path(self):
        return self._model_path

    def info(self):
        if not self._info:

            if not self._model_path and self._data:
                raise RuntimeError('tagger not trained, can not retrieve model information.')

            if not self._tagger:
                self.open(self._model_path)
            self._info = self._tagger.info()

        return self._info

    def print_top_trans(self, top=10):
        """
        Print top `top` most likely and unlikely transition between labels

        Parameters
        ----------
        top: int, positive integer
            The number of records to print
        """
        info = self.info()
        stats = Counter(info.transitions)
        print("Top %d most likely transitions:" % (top, ))
        print_transitions(stats.most_common(top))
        print("\nTop %d unlikely transitions:" % (top,))
        print_transitions(stats.most_common()[-1:-top-1:-1])

    def print_top_weighted_features(self, top=15):
        """
        Print top `top` most important and unimportant features.

        Parameters
        ----------
        top: int, positive integer
            The number of records to print
        """
        info = self.info()
        stats = Counter(info.state_features)
        print("Top %d positive features:" % (top,))
        print_state_features(stats.most_common(top))

        print("\nTop %d negative features:" % (top,))
        print_state_features(stats.most_common()[-1:-top-1:-1])


# basic named entity chunker tagger that does not use complex templates
class BasicNEChunkerTagger(CrfNeChunkerTaggerBase):

    def _feature_detect(self, sent):
        return self._sent2features(sent)

    # transform sentences to features
    def _word2features(self, sent, i):

        word = sent[i][0]
        postag = sent[i][1]

        features = [
            'bias',
            'word.lower=' + word.lower(),
            'length=' + str(len(word)),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
            'shape=' + get_shape(word),
            'type=' + get_type(word),
            'd&a=' + boolean(get_da(word))

        ]

        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.extend([
                '-1#word.lower=' + word1.lower(),
                '-1#word.istitle=%s' % word1.istitle(),
                '-1#word.isupper=%s' % word1.isupper(),
                '-1#postag=' + postag1,
                '-1#postag[:2]=' + postag1[:2],
                '-1#shape=' + get_shape(word1),
                '-1#type=' + get_type(word1),
                '-1#d&a=' + boolean(get_da(word1))
            ])
        else:
            features.append('__BOS__')

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.extend([
                '+1#word.lower=' + word1.lower(),
                '+1#word.istitle=%s' % word1.istitle(),
                '+1#word.isupper=%s' % word1.isupper(),
                '+1#postag=' + postag1,
                '+1#postag[:2]=' + postag1[:2],
                '+1#shape=' + get_shape(word1),
                '+1#type=' + get_type(word1),
                '+1#d&a=' + boolean(get_da(word1))
            ])
        else:
            features.append('__EOS__')

        return features

    def _sent2features(self, sent):
        return [self._word2features(sent, i) for i in range(len(sent))]


# NER chunker tagger using templates to extract complex features
class NEChunkerTagger(CrfNeChunkerTaggerBase):
    def __init__(self, name='model', train=None, model_path=None, templates=None):
        """
        Constructor for NEChunkerTagger.

        Parameters
        ----------
        templates: list of tuples.
            e.g. ('w', -1) means the feature of previous word,
                (('pos', 0), ('pos', 1)) means the concatenation of current word's POS tag and that of the next word's

        """
        super(NEChunkerTagger, self).__init__(name, train, model_path)
        self._templates = templates

    def _feature_detect(self, sent):
        # template-supported chunker tagger, `sent` must be ('w', 'pos', 'ne_chk') tuple list
        sent_dict = list(map(lambda x: {'w': x[0], 'pos': x[1], 'chk': x[2], 'F': []}, sent))
        feature_extractor(sent_dict, self._templates)
        return [x['F'] for x in sent_dict]