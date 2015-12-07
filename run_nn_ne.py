import sys, os
from numpy import *
from matplotlib.pyplot import *
import nn_ner.data_utils.utils as du
import nn_ner.data_utils.ner as ner
from utils.corpus import nn_vocab_path, nn_wv_path, wikiner_root
from nn_ner.nerwindow import WindowMLP
from optparse import OptionParser
from nn_ner.nerwindow import full_report, eval_performance
from utils.log import logger

if __name__ == '__main__':

    parser = OptionParser(usage='usage: python %prog [Options]')

    parser.add_option('-w', '--window', action='store', dest='window', default=3,
                      help='the window size to be used, not to be large, or the training will be extremely slow,'
                           'move to GPU NN library will be much much faster.')
    parser.add_option('-s', '--save', action='store', dest='save_path', default='test_pred.txt',
                      help='specified the file path to save prediction result, default: %default')
    fmts = ['eo', 'iob2', 'iob']
    parser.add_option('-f', '--format', action='store', dest='format', default='eo', type='choice', choices=fmts,
                      help='specified the data file format, default: %default, simplified NER tags.')
    parser.add_option('-p', '--portion', action='store', dest='portion', default=1.0,
                      help='train faster by only use part of training samples, within range of (0, 1.0]. default: %default.')
    parser.add_option('-c', '--corpus', action='store', dest='corpus', default='nn', type='choice',
                      choices=['wiki', 'nn'],
                      help='specified the corpus to trained and test, default: %default. the default one is small, train on wikiner '
                           'may take hours..')
    options, args = parser.parse_args()

    # seed the random number generator to make the result repeatable
    random.seed(10)
    # print random_weight_matrix(3,5)

    ratio = float(options.portion)
    if ratio > 1.0 or ratio <= 0.0:
        ratio = 1.0

    # Load the starter word vectors
    wv, word_to_num, num_to_word = ner.load_wv(nn_vocab_path,
                                               nn_wv_path)

    # string tag to integer label mapping
    IOB2 = ["O", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
    EO = ["O", "LOC", "MISC", "ORG", "PER"]

    tagnames = EO if options.format == 'eo' else IOB2
    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = du.invert_dict(num_to_tag)

    # Set window size
    windowsize = 3
    if int(options.window) > 0:
        windowsize = int(options.window)
    logger.info('setting window size to %d', windowsize)
    logger.info('using tagset: [%s]', ', '.join(tagnames))

    train_fp = None
    test_fp = None

    if options.corpus == 'nn':
        cwd = os.path.dirname(__file__)
        train_fp = os.path.join(cwd, 'data/nn/train')
        test_fp = os.path.join(cwd, 'data/nn/test')
        fields = 'w y'
    else:
        for p in os.listdir(wikiner_root):
            full_path = os.path.join(wikiner_root, p)
            if os.path.isfile(full_path) and p.find('small') >= 0:
                if p.find('train') >= 0:
                    train_fp = full_path
                elif p.find('test') >= 0:
                    test_fp = full_path
        fields = 'w pos y'

    if not train_fp:
        raise RuntimeError('train dataset is not found.')

    if not test_fp:
        raise RuntimeError('test dataset is not found.')

    need_conv = options.format == 'eo' and len(fields.split()) > 2
    # Load the training set
    docs = du.load_dataset(train_fp, fields, need_conv)
    X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                          wsize=windowsize)

    # Load the test set
    docs = du.load_dataset(test_fp, fields, need_conv)
    X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num,
                                        wsize=windowsize)

    train_sz = int(ratio * y_train.shape[0])
    test_sz = int(ratio * y_test.shape[0])
    logger.info('Total train dataset size: %d samples, in use: %d samples', y_train.shape[0], train_sz)
    logger.info('Total test dataset size: %d, in use: %d samples', y_test.shape[0], test_sz)

    nepoch = 5
    N = nepoch * y_train.shape[0]
    k = 5
    indices = range(train_sz)


    # A random schedule of N/k minibatches of size k, sampled with replacement from the training set.
    def idxiter_batches():
        num_batches = N / k
        for i in xrange(num_batches):
            yield random.choice(indices, k)


    # gradient check: passed it
    # clf.grad_check(X_train[0], y_train[0])

    clf = WindowMLP(wv, windowsize=windowsize, dims=[None, 100, len(tagnames)], reg=0.0001, alpha=0.01)
    clf.train_sgd(X_train[:train_sz], y_train[:train_sz], idxiter=idxiter_batches(), printevery=250000, costevery=50000)
    yp = clf.predict(X_test[:test_sz])
    full_report(y_test[:test_sz], yp, tagnames)  # full report, helpful diagnostics
    eval_performance(y_test[:test_sz], yp, tagnames)  # performance: optimize this F1

    logger.info('saving predictions to %s ...', options.save_path)
    ner.save_predictions(yp, options.save_path, tagnames)
    logger.info('done.')
