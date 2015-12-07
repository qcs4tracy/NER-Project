import re

"""
Example of template file:

# Unigram, [-1,0] means the second property(may be a postag) of the previous word
U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U04:%x[2,0]
U05:%x[-1,0]|%x[0,0]
U06:%x[0,0]|%x[1,0]

U10:%x[-2,1]
U11:%x[-1,1]
U12:%x[0,1]
U13:%x[1,1]
U14:%x[2,1]
U15:%x[-2,1]|%x[-1,1]
U16:%x[-1,1]|%x[0,1]
U17:%x[0,1]|%x[1,1]
U18:%x[1,1]|%x[2,1]

U20:%x[-2,1]|%x[-1,1]|%x[0,1]
U21:%x[-1,1]|%x[0,1]|%x[1,1]
U22:%x[0,1]|%x[1,1]|%x[2,1]

# Bigram
B

"""


class FeatureExtractor(object):
    """
    A feature extractor.
    """

    def __init__(self):
        self.macro = re.compile(r'%x\[(?P<row>[\d-]+),(?P<col>[\d]+)\]')
        self.inst = []
        self.t = 0
        self.templates = []

    def read(self, fi):
        # fi: str, the file path of the template file
        self.templates = []
        for line in fi:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('U'):
                self.templates.append(line.replace(':', '='))
            elif line == 'B':
                continue
            elif line.startswith('B'):
                raise StandardError('ERROR: bigram templates not supported: %s\n' % line)

    def _replace(self, m):
        # row offset
        row = self.t + int(m.group('row'))
        # column offset
        col = int(m.group('col'))
        # make the feature
        if row in range(0, len(self.inst)):
            return self.inst[row]['x'][col]
        else:
            return ''

    def apply(self, inst, t):
        self.inst = inst
        self.t = t
        for template in self.templates:
            f = re.sub(self.macro, self._replace, template)
            self.inst[t]['F'].append(f)

    def readiter(self, fi, sep=' '):
        X = []
        for line in fi:
            line = line.strip('\n')
            if not line:
                yield X
                X = []
            else:
                fields = line.split(sep)
                item = {
                    'x': fields[0:-1],
                    'y': fields[-1],
                    'F': []
                }
                X.append(item)


"""
raw python object templates for named eneity recognition (NER).
"""


def apply_templates(X, templates):
    """
    Generate features for an item sequence by applying feature templates.
    A feature template consists of a tuple of (name, offset) pairs,
    where name and offset specify a field name and offset from which
    the template extracts a feature value. Generated features are stored
    in the 'F' field of each item in the sequence.
    @type   X:      list of mapping objects
    @param  X:      The item sequence.
    @type   template:   tuple of (str, int)
    @param  template:   The feature template.
    """
    n = len(X)
    for template in templates:
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        for t in range(len(X)):
            values = []
            for field, offset in template:
                p = t + offset
                if p < 0 or p >= n:
                    values = []
                    break
                values.append(X[p][field])
            if values:
                X[t]['F'].append('%s=%s' % (name, '|'.join(values)))


# Separator of field values.
separator = ' '

# Field names of the input data.
fields = 'w pos chk'

# get the shape encoding of token
def get_shape(token):
    r = ''
    for c in token:
        if c.isupper():
            r += 'U'
        elif c.islower():
            r += 'L'
        elif c.isdigit():
            r += 'D'
        elif c in ('.', ','):
            r += '.'
        elif c in (';', ':', '?', '!'):
            r += ';'
        elif c in ('+', '-', '*', '/', '=', '|', '_'):
            r += '-'
        elif c in ('(', '{', '[', '<'):
            r += '('
        elif c in (')', '}', ']', '>'):
            r += ')'
        else:
            r += c
    return r

# duplicate consecutive chars in src
def degenerate(src):
    dst = ''
    for c in src:
        if not dst or dst[-1] != c:
            dst += c
    return dst


# get the type of a words, all types defined in T below.
def get_type(token):

    T = (
        'AllUpper', 'AllDigit', 'AllSymbol',
        'AllUpperDigit', 'AllUpperSymbol', 'AllDigitSymbol',
        'AllUpperDigitSymbol',
        'InitUpper',
        'AllLetter',
        'AllAlnum',
    )

    all = set(T)

    if not token:
        return 'EMPTY'

    for idx in range(len(token)):
        c = token[idx]
        if c.isupper():
            all.discard('AllDigit')
            all.discard('AllSymbol')
            all.discard('AllDigitSymbol')
        elif c.isdigit() or c in (',', '.'):
            all.discard('AllUpper')
            all.discard('AllSymbol')
            all.discard('AllUpperSymbol')
            all.discard('AllLetter')
        elif c.islower():
            all.discard('AllUpper')
            all.discard('AllDigit')
            all.discard('AllSymbol')
            all.discard('AllUpperDigit')
            all.discard('AllUpperSymbol')
            all.discard('AllDigitSymbol')
            all.discard('AllUpperDigitSymbol')
        else:
            all.discard('AllUpper')
            all.discard('AllDigit')
            all.discard('AllUpperDigit')
            all.discard('AllLetter')
            all.discard('AllAlnum')

        if idx == 0 and not c.isupper():
            all.discard('InitUpper')

    if len(all) > 0:
        return all.pop()

    return 'Other'


def get_2d(token):
    return len(token) == 2 and token.isdigit()


def get_4d(token):
    return len(token) == 4 and token.isdigit()


def get_da(token):
    bd = False
    ba = False
    for c in token:
        if c.isdigit():
            bd = True
        elif c.isalpha():
            ba = True
        else:
            return False
    return bd and ba


def get_dand(token, p):
    bd = False
    bdd = False
    for c in token:
        if c.isdigit():
            bd = True
        elif c == p:
            bdd = True
        else:
            return False
    return bd and bdd


def get_all_other(token):
    for c in token:
        if c.isalnum():
            return False
    return True


def get_capperiod(token):
    return len(token) == 2 and token[0].isupper() and token[1] == '.'


def contains_upper(token):
    b = False
    for c in token:
        b |= c.isupper()
    return b


def contains_lower(token):
    b = False
    for c in token:
        b |= c.islower()
    return b


def contains_alpha(token):
    b = False
    for c in token:
        b |= c.isalpha()
    return b


def contains_digit(token):
    b = False
    for c in token:
        b |= c.isdigit()
    return b


def contains_symbol(token):
    b = False
    for c in token:
        b |= ~c.isalnum()
    return b


def boolean(v):
    return 'yes' if v else 'no'


def observation(v, defval=''):
    # Lowercased token.
    v['wl'] = v['w'].lower()
    # Token shape.
    v['shape'] = get_shape(v['w'])
    # Token shape degenerated.
    v['shaped'] = degenerate(v['shape'])
    # Token type.
    v['type'] = get_type(v['w'])

    # Prefixes (length between one to four).
    v['p1'] = v['w'][0] if len(v['w']) >= 1 else defval
    v['p2'] = v['w'][:2] if len(v['w']) >= 2 else defval
    v['p3'] = v['w'][:3] if len(v['w']) >= 3 else defval
    v['p4'] = v['w'][:4] if len(v['w']) >= 4 else defval

    # Suffixes (length between one to four).
    v['s1'] = v['w'][-1] if len(v['w']) >= 1 else defval
    v['s2'] = v['w'][-2:] if len(v['w']) >= 2 else defval
    v['s3'] = v['w'][-3:] if len(v['w']) >= 3 else defval
    v['s4'] = v['w'][-4:] if len(v['w']) >= 4 else defval

    # Two digits
    v['2d'] = boolean(get_2d(v['w']))
    # Four digits.
    v['4d'] = boolean(get_4d(v['w']))
    # Alphanumeric token.
    v['d&a'] = boolean(get_da(v['w']))
    # Digits and '-'.
    v['d&-'] = boolean(get_dand(v['w'], '-'))
    # Digits and '/'.
    v['d&/'] = boolean(get_dand(v['w'], '/'))
    # Digits and ','.
    v['d&,'] = boolean(get_dand(v['w'], ','))
    # Digits and '.'.
    v['d&.'] = boolean(get_dand(v['w'], '.'))
    # A uppercase letter followed by '.'
    v['up'] = boolean(get_capperiod(v['w']))

    # An initial uppercase letter.
    v['iu'] = boolean(v['w'] and v['w'][0].isupper())
    # All uppercase letters.
    v['au'] = boolean(v['w'].isupper())
    # All lowercase letters.
    v['al'] = boolean(v['w'].islower())
    # All digit letters.
    v['ad'] = boolean(v['w'].isdigit())
    # All other (non-alphanumeric) letters.
    v['ao'] = boolean(get_all_other(v['w']))

    # Contains a uppercase letter.
    v['cu'] = boolean(contains_upper(v['w']))
    # Contains a lowercase letter.
    v['cl'] = boolean(contains_lower(v['w']))
    # Contains a alphabet letter.
    v['ca'] = boolean(contains_alpha(v['w']))
    # Contains a digit.
    v['cd'] = boolean(contains_digit(v['w']))
    # Contains a symbol.
    v['cs'] = boolean(contains_symbol(v['w']))


# unigram feature
U = [
    'w', 'wl', 'pos', 'shape', 'shaped', 'type',
    'p1', 'p2', 'p3', 'p4',
    's1', 's2', 's3', 's4',
    '2d', '4d', 'd&a', 'd&-', 'd&/', 'd&,', 'd&.', 'up',
    'iu', 'au', 'al', 'ad', 'ao',
    'cu', 'cl', 'ca', 'cd', 'cs',
]

# bigram feature
B = ['w', 'pos', 'shaped', 'type']

templates = []
for name in U:
    templates += [((name, i),) for i in range(-2, 3)]
for name in B:
    templates += [((name, i), (name, i + 1)) for i in range(-2, 2)]


def feature_extractor(X, tmpl=None):

    # Append observations.
    for x in X:
        observation(x)

    if not tmpl:
        # Apply the feature templates.
        apply_templates(X, templates)
    else:
        apply_templates(X, tmpl)

    # Append begin of sentence and end of stence features.
    if X:
        X[0]['F'].append('__BOS__')
        X[-1]['F'].append('__EOS__')