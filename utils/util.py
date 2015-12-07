__author__ = 'qiuchusheng'
import os, random
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

def train_test_split(fp, exact=True, ratio=0.2):
    """
    Randomly split the file into two file splits.

    Parameters
    ----------
    fp: str
        The file path of the data to split.

    exact: bool
        Approximate split or accurate split

    ratio: float
        Test data size to trian data size ratio.
    """
    if not os.path.isfile(fp):
        raise IOError("%s is not a file." % (fp,))
    # if the file is small, we can just shuffle all lines and split
    train_fn = "%s-train" % (fp,)
    test_fn = "%s-test" % (fp,)
    with open(fp) as f:
        if exact:
            data = f.readlines()
            sz = len(data)
            random.shuffle(data)
            test_sz = int(sz * ratio)
            with open(test_fn, 'w') as tef:
                tef.writelines(data[:test_sz])
            with open(train_fn, 'w') as trf:
                trf.writelines(data[test_sz:])
        else:
            pred = lambda: random.random() >= ratio
            with open(test_fn, 'w') as tef, open(train_fn, 'w') as trf:
                for line in f.readlines():
                    if pred():
                        trf.write(line)
                    else:
                        tef.write(line)
    return (train_fn, test_fn)

# The IOB2 tags
O = 'O'

B = {'LOC': 'B-LOC',
     'PER': 'B-PER',
     'ORG': 'B-ORG',
     'MISC': 'B-MISC'}

I = {'LOC': 'I-LOC',
     'PER': 'I-PER',
     'ORG': 'I-ORG',
     'MISC': 'I-MISC'}


def wikiner2iob2(sent):
    """
    Parameters
    ----------
    sent: str
        One line of wiki ner data
    """
    # word|pos|ner-chunk
    wpn = sent.strip()
    if not wpn:  ##empty line
        return ''
    wpn = wpn.split()
    tagged_sent = filter(lambda x: len(x) == 3, map(lambda x: x.split('|'), wpn))
    inchk = False
    n = len(tagged_sent)
    for i in range(n):
        ne_tag = tagged_sent[i][-1]
        if ne_tag == O:
            inchk = False
            continue
        else:
            prefix, cat = ne_tag.split('-')
            if prefix == 'I' and not inchk:
                tagged_sent[i][-1] = B[cat]
            inchk = True
    return '\n'.join(map(lambda x: ' '.join(x), tagged_sent)) + '\n\n'





class Wikiner2ConllConverter(object):
    """
    This class is used to convert wiki ner file to conll2002-3 format in order to use
    the full functionality of nltk.corpus.ConllCorpusReader.
    """

    def __init__(self, source_file, dest_filename=None):
        """
        Parameters
        ----------
        source_file: str
            The file path of wiki ner file.

        dest_filename: str
            The file path to write converted file to.
        """
        self._source_file = source_file
        if dest_filename:
            self._dest_file = dest_filename
        else:
            self._dest_file = "%s-conll-fmt" % (source_file,)
        if not os.path.isfile(self._source_file):
            raise IOError('the source file %s specified is not exist.' % (source_file,))
        if os.path.isfile(self._dest_file):
            pass
            # raise RuntimeWarning('the output file: %s is already exist, try another path.' % (self._dest_file, ))

    def _tagged_to_conll(self, tagged_word):
        if (len(tagged_word) != 3):
            raise ValueError('Inconsistent data: tagged words with more than 3 columns!')
        return " ".join(tagged_word) + "\n"

    def convert(self):
        with open(self._source_file, 'r') as f:
            with open(self._dest_file, 'w') as out:
                for sent in f.readlines():
                    # word|pos|ner-chunk
                    wpn = sent.strip()
                    if not wpn:  ##empty line
                        continue
                    wpn = wpn.split()
                    tagged_sent = list(map(lambda x: x.split('|'), wpn))
                    lines = list(map(self._tagged_to_conll, tagged_sent))
                    out.writelines(lines)
                    out.write("\n")
                    # logging can goes here


# convert IOB format to IOB2 format, which is used by Conll dataset
def iob_to_iob2(fp, dest_fp):
    """
    Convert a IOB ner data from Conll2002-3 to a IOB2 file format.
    """

    if not os.path.isfile(fp):
        raise IOError('file %s not exist or is not a file.' % (fp,))

    in_chk = False
    # process each line and write to output
    with open(fp, 'r') as fin, open(dest_fp, 'w') as fout:
        for i, line in enumerate(fin.readlines()):

            line = line.strip()

            if line == '':  # sentence boundary, write a newline
                fout.write('\n')
                in_chk = False
                continue

            fields = line.split()
            if (len(fields) != 3):
                raise ValueError("line %d is not consistent: %s" % (i, line))

            ne_tag = fields[-1]
            if (ne_tag == O):
                fout.write(line + '\n')
                in_chk = False
            else:
                prefix, cat = ne_tag.split('-')
                if prefix == 'B':
                    fout.write(line + '\n')
                else:  # prefix == 'I'
                    if not in_chk:
                        fields[-1] = B[cat]
                        fout.write(' '.join(fields) + '\n')
                    else:
                        fout.write(line + '\n')
                in_chk = True


# prepare the data for use
def prepare_ner_data(fp, dest_folder, test_train_ratio=0.2):
    if not os.path.isdir(dest_folder):
        raise IOError('%s is not a directory.' % (dest_folder,))

    train_fp, test_fp = train_test_split(fp, ratio=test_train_ratio)
    tr_dest = os.path.join(dest_folder, os.path.basename(train_fp) + '.tmp')
    te_dest = os.path.join(dest_folder, os.path.basename(test_fp) + '.tmp')
    Wikiner2ConllConverter(train_fp, tr_dest).convert()
    Wikiner2ConllConverter(test_fp, te_dest).convert()

    # now, convert the IOB foramt to IOB2 format
    iob_to_iob2(tr_dest, tr_dest[:-4] + '-iob2')
    iob_to_iob2(te_dest, te_dest[:-4] + '-iob2')

    #clean the tmp files
    try:
        os.remove(train_fp)
        os.remove(test_fp)
        os.remove(tr_dest)
        os.remove(te_dest)
    except OSError:
        pass


# fetch specified number of lines to another file, wikiner has each line for a sentence.
# so this is actually extract specified number of records from the dataset
def fetch_lines(fp, dest_fp, n=500):
    with open(fp, 'r') as f, open(dest_fp, 'w') as out:
        i = 0
        while i < n:
            line = f.readline()
            if line == '':
                # EOF
                break
            if line.strip() != '':
                out.write(line)
                i += 1


def ner_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )