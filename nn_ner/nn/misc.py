##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):

    c = sqrt(6) / sqrt(m + n)
    A0 = random.uniform(-c, c, (m, n))

    assert(A0.shape == (m,n))
    return A0