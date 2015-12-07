from __future__ import absolute_import
from nltk.corpus.reader import ConllChunkCorpusReader
from . import corpus
from .corpus import corpus_root
import gc, os
import nltk

def _make_bound_method(func, self):
    """
    Creating bound methods (used for _unload).
    """
    class Foo(object):
        def meth(self): pass
    f = Foo()
    bound_method = type(f.meth)

    try:
        return bound_method(func, self, self.__class__)
    except TypeError: # python3
        return bound_method(func, self)

class LazyCorpusLoader(object):

    def __init__(self, name, reader_cls, *args, **kwargs):
        from nltk.corpus.reader.api import CorpusReader
        assert issubclass(reader_cls, CorpusReader)
        self.__name = self.__name__ = name
        self.__reader_cls = reader_cls
        self.__args = args
        self.__kwargs = kwargs

    def __load(self):
        try:
            root = nltk.data.find(self.__name, [corpus_root])
        except LookupError:
            raise LookupError('this corpus does not exist in path %s' %
                              (os.path.join(corpus_root, self.__name)))

        # Load the corpus.
        corpus = self.__reader_cls(root, *self.__args, **self.__kwargs)

        # Transform ourselves into the corpus by modifying our own __dict__ and
        # __class__ to match that of the corpus.

        args, kwargs = self.__args, self.__kwargs
        name, reader_cls = self.__name, self.__reader_cls

        self.__dict__ = corpus.__dict__
        self.__class__ = corpus.__class__

        # _unload support: assign __dict__ and __class__ back, then do GC.
        # after reassigning __dict__ there shouldn't be any references to
        # corpus data so the memory should be deallocated after gc.collect()
        def _unload(self):
            lazy_reader = LazyCorpusLoader(name, reader_cls, *args, **kwargs)
            self.__dict__ = lazy_reader.__dict__
            self.__class__ = lazy_reader.__class__
            gc.collect()

        self._unload = _make_bound_method(_unload, self)

    def __getattr__(self, attr):

        if attr == '__bases__':
            raise AttributeError("LazyCorpusLoader object has no attribute '__bases__'")

        self.__load()

        # __load() changes __class__ to corpus reader,
        # so getattr() will be looked up using new class.
        return getattr(self, attr)

    def __repr__(self):
        return '<%s in %r (not loaded yet)>' % (
            self.__reader_cls.__name__, self.__name)

    def _unload(self):
        pass


wikiner_tagset = ('LOC', 'PER', 'ORG', 'MISC')
wikiner_corpus = LazyCorpusLoader('wikiner', ConllChunkCorpusReader, corpus.corpus_pattn, corpus, wikiner_tagset, encoding='utf-8')
wikiner_small_corpus = LazyCorpusLoader('wikiner', ConllChunkCorpusReader, corpus.small_corpus_pattn, wikiner_tagset, encoding='utf-8')
