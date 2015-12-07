from utils.data import wikiner_small_corpus, wikiner_corpus
from crf_ner.crf_ner_chunk import BasicNEChunkerTagger, NEChunkerTagger, default_train_params, sent2labels
from utils.util import ner_classification_report
from optparse import OptionParser
from utils.log import logger
import itertools, os

if __name__ == '__main__':

    model_root = os.path.join(os.path.dirname(__file__), 'crf_ner', 'model')

    parser = OptionParser(usage='usage: python %prog [Options]')
    parser.add_option('-m', '--model', action='store', dest='model_path',
                      default='', help='the path to the model file that is to be opened.')
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False,
                      help='only valid when -m not specified, -v option means print training details.: %default')
    parser.add_option('-s', '--nosmall', action='store_false', dest='use_small', default=True,
                      help='specified -s or --nosmall means that small dataset will be created for fast training and testing, '
                           +'default: %default. the small dataset will be 0.3 the size of original one')
    parser.add_option('-b', '--nobasic', action='store_false', dest='use_basic', default=True,
                      help='use NER tagger that supports basic feature detector, if not use it, a complex' +
                           'model with template defined feature detector will be used.')
    options, args = parser.parse_args()

    all_fileids = None
    datatset = None
    test_fileids = []
    train_fileids = []

    if options.use_small:
        datatset = wikiner_small_corpus
    else:
        datatset = wikiner_corpus

    all_fileids = datatset.fileids()
    logger.info('\n\tLoading chosen data files: \n\t\t%s' %('\n\t\t'.join(all_fileids)))
    for fid in all_fileids:
        if fid.count('test') > 0:
            test_fileids.append(fid)
        elif fid.count('train') > 0:
            train_fileids.append(fid)

    if len(test_fileids) == 0 or len(train_fileids) == 0:
        raise RuntimeError('there must be at least one dataset for both train and test, %d'
                           + 'train file and %d test file found.' % (len(train_fileids), len(test_fileids)))



    train_sents = None
    test_sents = datatset.iob_sents(test_fileids)
    tagger_class = None
    ner_tagger = None
    tagger_name = ''

    if options.use_basic:
        tagger_class = BasicNEChunkerTagger
        tagger_name = 'basic-ner-tagger'
    else:
        tagger_class = NEChunkerTagger
        tagger_name = 'ner-tagger'

    if options.model_path != '':
        ner_tagger = tagger_class(tagger_name, train=train_sents, model_path=options.model_path)
    else:
        train_sents = datatset.iob_sents(train_fileids)
        ner_tagger = tagger_class(tagger_name, train=train_sents, model_path=None)
        logger.info('Begin training ...')
        ner_tagger.train(options.verbose, default_train_params, max_iterations=500)

    logger.info('Begin tagging test sentences...')
    y_pred = itertools.imap(ner_tagger.tag, test_sents)
    y_true = itertools.imap(sent2labels, test_sents)

    logger.info('Begin evaluating...')
    report = ner_classification_report(y_true, y_pred)
    # print the report
    print('\n\nEvaluation Report:\n')
    print(report)
    print('\n{:^35}\n'.format('====model statistics===='))
    ner_tagger.print_top_trans()
    print('\n')
    ner_tagger.print_top_weighted_features()

    if options.model_path == '':
        fn = os.path.basename(ner_tagger.model_path())
        dest = os.path.join(model_root, fn)
        logger.info('saving model to %s', dest, 'crfsuite')
        ner_tagger.save(model_root)
        # clean the
        fmt = '\n** NOTE that you can directly load model and evaluate test set by adding `--model=%s`\n'\
              + 'or `-m %s` option, but it must be consist with the tagger model you choose\n'\
              + '. i.e. basic tagger and template tagger model file must be distinguished.'
        print(fmt % (dest, dest))
