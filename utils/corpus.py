import bz2
import os
import sys
import tarfile
import urllib2
import zipfile
from optparse import OptionParser
from util import prepare_ner_data
from log import logger


"""
This script is used to download and preprocess the dataset.
"""

dir_name = os.path.dirname
dataset_url = 'http://sydney.edu.au/engineering/it/~joel/wikiner/aij-wikiner-en-wp3.bz2'
corpus_root = os.path.join(dir_name(dir_name(os.path.abspath(__file__))), 'data')
corpus_pattn = '.*wp3\-(train|test)\-.*'
small_corpus_pattn = '.*\-small\-(train|test)\-.*'
wikiner_root = os.path.join(corpus_root, 'wikiner')
corpus_path = os.path.join(wikiner_root, os.path.splitext(os.path.basename(dataset_url))[0])

nn_data__url = 'http://104.131.6.249:8888/files/qiu/data/vocab-wv.zip'
nn_root = os.path.join(corpus_root, 'nn')
nn_vocab_path = os.path.join(nn_root, 'vocab')
nn_wv_path = os.path.join(nn_root, 'wordVectors')


# download the corpus
def download_corpus(url, root='.', fmt='bz2', rename=None):
    _BLOCK_SIZE = 1024 << 2
    response = urllib2.urlopen(url)
    if not rename:
        fp = os.path.join(root, os.path.splitext(os.path.basename(url))[0])
    else:
        fp = os.path.join(root, rename)

    if response.code != 200:
        raise IOError('download failed due to network: response code[%d]', response.code)

    total_length = response.headers.getheader('Content-Length', None)
    total_length = int(total_length) if total_length != None else None
    dwn_len = 0

    # with open(fp, 'a') as fo:
    #     fo.write(bz2.decompress(response.read()))

    def begin():
        logger.info("Start downloading dataset from %s" % (url,))

    def progress_bar(dl, tl):
        if tl == None:
            return
        done = int(50 * dl / tl)
        sys.stdout.write("\r[%s>%s]: %d%%" % ('=' * done, ' ' * (50 - done), done * 2))

    def done():
        print('\n')
        logger.info('Downloaded Finished: \n\tthe downloaded dataset is placed at %s\n', fp)

    def read_in_blocks(file_obj, block_size):
        while True:
            data = file_obj.read(block_size)
            if not data:
                raise StopIteration()
            else:
                yield data

    post_process = {
        'plain': lambda x: x,  # identity function, no change
        'bz2': bz2.BZ2Decompressor().decompress
    }

    data_post_proc = post_process[fmt]
    with open(fp, 'w') as fo:
        begin()
        for data in read_in_blocks(response, _BLOCK_SIZE):
            dwn_len += len(data)
            progress_bar(dwn_len, total_length)
            fo.write(data_post_proc(data))
        done()
    response.close()
    return fp


def extract_all(fp, fmt=None, root='.'):
    def find_decompressor(fmt_):
        dc_map = {
            'gzip': (tarfile.open, 'r:gz'),
            'zip': (zipfile.ZipFile, 'r'),
            'bz2': (tarfile.open, 'r:bz2')
        }
        if not fmt_ in dc_map:
            raise ValueError('the `%s` format is not supported.' % (fmt_,))
        return dc_map[fmt_]

    if not fmt:
        if fp.endswith('.zip'):
            fmt = 'zip'
        elif fp.endswith('.tar.gz') or fp.endswith('.tgz'):
            fmt = 'gzip'
        elif fp.endswith('.tar.bz2') or fp.endswith('.tbz'):
            fmt = 'bz2'
        else:
            fmt = 'None'

    opener, mode = find_decompressor(fmt)

    try:
        with opener(fp, mode) as f:
            f.extractall(path=root)
    except StandardError as e:
        logger.fatal('error occur while extracting file %s: %s', fp, str(e))


def download_split(ratio=0.2, force_dwl=False):
    # if not exist make the directory
    if not os.path.exists(wikiner_root) or not os.path.isdir(wikiner_root):
        os.mkdir(wikiner_root)

    split = float(ratio)
    if split >= 1.0 or split <= 0.0:
        raise ValueError('invalid ratio specified, need 0.0 < ratio < 1.0, %.2f given.' % (split))

    # if not already downloaded, go and download it
    if not os.path.isfile(corpus_path) or force_dwl:
        download_corpus(dataset_url, wikiner_root)
        print('Begin splitting dataset into training set and test set...')
        prepare_ner_data(corpus_path, wikiner_root, split)
    else:
        print('dataset `%s` is already downloaded.' % (corpus_path,))


if __name__ == '__main__':

    parser = OptionParser(usage='usage: python %prog [Options]')
    parser.add_option('-r', '--ratio', action='store', dest='ratio',
                      default=0.2, help='the test to train data split ratio, default: %default')
    parser.add_option('-f', '--force', action='store_true', dest='force', default=False,
                      help='force the download of dataset regardless of its existence, default: %default')
    parser.add_option('-s', '--small', action='store_true', dest='small', default=False,
                      help='specified -s means that small dataset will be created for fast training and testing, '
                           + 'default: %default. the small dataset will be 0.3 the size of original one')

    options, args = parser.parse_args()
    download_split(float(options.ratio), options.force)

    if options.small:

        name = os.path.basename(corpus_path)
        small_test = os.path.join(wikiner_root, '%s-small-test-iob2' % (name,))
        small_train = os.path.join(wikiner_root, '%s-small-train-iob2' % (name,))

        if not (os.path.exists(small_test) and os.path.exists(small_train)) or options.force:
            import random
            from util import wikiner2iob2

            small_ratio = 0.3
            # if random() <= test_prob then place the sentence to small test set
            # if random() >= train_prob then place the sentence to small train set
            test_prob = small_ratio * options.ratio
            train_prob = 1.0 - (1 - options.ratio) * small_ratio

            with open(corpus_path, 'r') as f, open(small_train, 'w') as str, open(small_test, 'w') as ste:
                for line in f.readlines():
                    r = random.random()
                    if r <= test_prob:
                        ste.write(wikiner2iob2(line))
                    elif r >= train_prob:
                        str.write(wikiner2iob2(line))
        else:
            print('small dataset %s & %s already exist.' % (small_train, small_train))

    if not os.path.isdir(nn_root):
        os.mkdir(nn_root)

    if not (os.path.isfile(nn_vocab_path) and os.path.isfile(nn_wv_path)) or options.force:
        tmpf = download_corpus(nn_data__url, nn_root, 'plain')
        extract_all(tmpf, fmt='zip', root=nn_root)

        # strip off file extension
        for fn in os.listdir(nn_root):
            fext = os.path.splitext(fn)
            if fext[1] != '':
                os.rename(os.path.join(nn_root, fn), os.path.join(nn_root, fext[0]))

        # clean the downloaded file
        try:
            os.remove(tmpf)
        except IOError:
            pass

    print('available data files: \n\t%s\n' % '\n\t'.join(os.listdir(wikiner_root)))
    print('auxiliary data files: \n\t%s' % '\n\t'.join(os.listdir(nn_root)))