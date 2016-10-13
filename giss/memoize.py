import functools
from giss import checksum

"""Generalized functional-style access to data."""

# https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize



#class memoized(object):
#    def __init__(self, *args, **kwargs):
#        self.args = args
#        self.kwargs = kwargs    # dict_fn, id_fn
#
#    def __call__(self, func):
#        return _memoized(func, *self.args, **self.kwargs)


def arg_decorator(decorator_fn):
    """Meta-decorator that makes it easier to write decorators taking args.
    See:
       * old way: http://scottlobdell.me/2015/04/decorators-arguments-python
       * new way: memoized below (no lambdas required)"""

    class real_decorator(object):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs    # dict_fn, id_fn

        def __call__(self, func):
            return decorator_fn(func, *self.args, **self.kwargs)

    return real_decorator

class memoized(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs    # dict_fn, id_fn

    def __call__(self, func):
        return _memoized(func, *self.args, **self.kwargs)



#@arg_decorator
class _memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    NOTE: This needs to be a class to keep it checksummable"""

    def __init__(self, func, dict_fn=dict, id_fn=checksum.checksum):
        self.func = func
        self.dict_fn = dict_fn
        self.id_fn = id_fn

    def __call__(self, *args, **kwargs):
        # First argument of a memoized function is always the memoization state
        # It's a dict(function --> cache-for-that-function)
        ms = args[0]

        # Look up the dict used for memoizing this function
        func_id = self.id_fn(self.func)
        try:
            cache = ms[func_id]
        except KeyError:
            cache = self.dict_fn()
            ms[func_id] = cache

        args_id = self.id_fn((args[1:], kwargs))

        if args_id not in cache:
            value = self.func(*args, **kwargs)
            cache[args_id] = value

        return cache[args_id]


    # Allow to decorate methods as well as functions
    # See: http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        new_func = self.func.__get__(obj, type)
        return self.__class__(new_func)

    # Enable checksums on this class
    hash_version=1
    def hashup(self, hash):
        checksum.hashup(hash, self.func)


class xxmemoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    NOTE: This needs to be a class to keep it checksummable"""

    def __init__(self, dict_fn=dict, id_fn=checksum.checksum):
        """dict_fn:
            Function to create a new dict in the cacheset, if one does
            not already exist for a given function."""
#        self.func = func
#        print('self.func',self.func)
        self.dict_fn = dict_fn
        self.id_fn = id_fn

    def __call__(self, *args, **kwargs):
        # First argument of a memoized function is always the memoization state
        # It's a dict(function --> cache-for-that-function)
        print('******b', args)
        ms = args[0]

        # Look up the dict used for memoizing this function
        print('******', args)
        func_id = self.id_fn(self.func)
        try:
            cache = ms[func_id]
        except KeyError:
            cache = self.dict_fn()
            ms[func_id] = cache

        args_id = self.id_fn((args, kwargs))
        if args_id not in cache:
            value = self.func(*args, **kwargs)
            self.cache[args_id] = value
        return cache[args_id]



#    def __repr__(self):
#        '''Return the function's docstring.'''
#        return self.func.__doc__
#
#   def __get__(self, obj, objtype):
#      '''Support instance methods.'''
#      return functools.partial(self.__call__, obj)
