##
# Utility functions for NER assignment
# Assigment 2, part 1 for CS224D
##

from utils import invert_dict
from numpy import *

def load_wv(vocabfile, wvfile):
    wv = loadtxt(wvfile, dtype=float)
    with open(vocabfile) as fd:
        words = [line.strip() for line in fd]
    num_to_word = dict(enumerate(words))
    word_to_num = invert_dict(num_to_word)
    return wv, word_to_num, num_to_word


def save_predictions(y, filename, tagset=None):
    """Save predictions, one per line."""
    with open(filename, 'w') as fd:
        if tagset:
            fd.write("TAG: ")
            fd.write(" ".join(tagset))
            fd.write("\nTOTAL: %s\n" % (len(y), ))
        fd.write("\n".join(map(str, y)))
        fd.write("\n")