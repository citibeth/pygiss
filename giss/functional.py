import types
from giss import giutil
from giss.checksum import hashup
from giss import giutil, gicollections
import types
import inspect
import operator


# --------------------------------------------------------
def arg_decorator(decorator_fn):
    """Meta-decorator that makes it easier to write decorators taking args.
    See:
       * old way: http://scottlobdell.me/2015/04/decorators-arguments-python
       * new way: See memoize.files (no lambdas required)"""

    class real_decorator(object):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs    # dict_fn, id_fn

        def __call__(self, func):
            return decorator_fn(func, *self.args, **self.kwargs)

    return real_decorator

# ---------------------------------------------------------
# Universal for all Functions...
#    (f + g)(x) = f(x) + g(x)
class Function(object):
    def __add__(self, other):
        return lift(operator.add, self, other)
    def __multiply__(self, other):
        return lift(operator.multiply, self, other)
class lift(Function):
    """Turns a function on values into a function on functions."""
    def __init__(self, lifted_fn, *funcs):
        self.lifted_fn = lifted_fn
        self.funcs = funcs
    def __call__(self, *args, **kwargs):
        args = tuple(fn(*args, **kwargs) for fn in self.funcs)
        return self.lifted_fn(*args)
    def __repr__(self):
        return '{}({})'.format(self.lifted_fn, ','.join(repr(x) for x in self.funcs))

# ---------------------------------------------------------
# Partial binding of functions

class _arg(object):
    """Tagging class"""
    def __init__(self, index):
        self.index = index
    def __repr__(self):
        return '_arg({})'.format(self.index)

class BoundFunction(Function):
    """Reorder positional arguments.
    Eg: g = f('yp', _1, 17, _0, dp=23)
    Then g('a', 'b', another=55) --> f('yp', 'b', 17, 'a', dp=23, another=55)

    TODO:
       1. When wrapping multiple _Binds, create just a single level???
       2. Get kwargs working
    """

    def __init__(self, fn, *bound_args, **bound_kwargs):
        # Maximum index referred to by the user.
        # Inputs to f above this index will be passed through
        self.fn = fn
        self.bound_args = bound_args
        self.bound_kwargs = bound_kwargs
        self.first_unbound = 1+max(
            (x.index if isinstance(x, _arg) else -1 for x in bound_args),
            default=-1)

    def __call__(self, *gargs, **gkwargs):
        fargs = \
            [gargs[x.index] if isinstance(x, _arg) else x
                for x in self.bound_args] + \
            list(gargs[self.first_unbound:])

        fkwargs = dict(self.bound_kwargs)
        fkwargs.update(gkwargs)    # Overwrite keys
        return self.fn(*fargs, **fkwargs)

    hash_version=0
    def hashup(self,hash):
        hashup(hash, (self.fn, self.bound_args, self.bound_kwargs))
    def __repr__(self):
        return 'bind({}, {}, {})'.format(self.fn, self.bound_args, self.bound_kwargs)

def bind(fn, *bound_args, **bound_kwargs):
    if False and isinstance(fn, BoundFunction):
        # (Don't bother with this optimization for now...)
        # Re-work bound args...
        pass
    elif isinstance(fn, _Tuple):
        # Bind inside the Tuple, to retain tuple nature of our function
        return fn.construct(bind(x, *bound_args, **bound_kwargs) for x in fn)
    else:
        return BoundFunction(fn, *bound_args, **bound_kwargs)


def function(python_fn):
    """Decorator Wraps a Python function into a Function."""
    return bind(python_fn)
# ---------------------------------------------------------
# Good for summing >1 functions

class sum(Function):
    def __init__(self, *funcs):
        self.funcs = funcs
    def __call__(self, *args, **kwargs):
        sum = funcs[0](*args, **kwargs)
        for fn in funcs[1:]:
            sum += fn(*args, **kwargs)
        return sum

class product(Function):
    def __init__(self, *funcs):
        self.funcs = funcs
    def __call__(self, *args, **kwargs):
        product = funcs[0](*args, **kwargs)
        for fn in funcs[1:]:
            product *= fn(*args, **kwargs)
        return product
# ---------------------------------------------------------
class thunkify(Function):
    """Decorator that replaces a function with a thunk constructor.
    When called, the resulting thunk will run the original function.

    Suppose f :: X -> Y         # Haskell notation
    Then thunkify f x :: Y
    or    thunkfiy :: Function -> X -> Y
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return bind(self.fn, *args, **kwargs)

    def __repr__(self):
        return 'thunkify({})'.format(self.fn)
# -------------------------------------------------------
class Wrap(Function):
    """A thunk that wraps a value; calling the thunk will return the value.
    This serves as a base class to define arithmetic operations on (wrapped) values
    when combining functions."""
    def __init__(self, value):
        self.value = value
    def __call__(self):
        return self.value
    def __repr__(self):
        return 'Wrap({})'.format(self.value)

# -------------------------------------------------------------
class WrapCombine(Wrap):
    """A thunk that wraps a value; calling the thunk will return the value.
    All operations are mapped to a supplied "combine" function."""
    def __init__(self, value, combine_fn):
        self.value = value
        self.combine_fn = combine_fn
    def __add__(self, other):
        return WrapCombine(self.combine_fn(self.value, other.value), self.combine_fn)

# -------------------------------------------------------------
def intersect_dicts(a,b):
    """Combine function: Returns only entries with the same value in both dicts."""
    return {key : a[key] \
        for key in a.keys()&b.keys() \
        if a[key] == b[key]}

def none_if_different(a,b):
    """Combine function: Keep only if things are the same."""
    return a if a==b else None
# -------------------------------------------------------------
class _Tuple(Function):
    """Avoid problems of multiple inheritence and __init__() methods.
    See for another possible soultion:
    http://stackoverflow.com/questions/1565374/subclassing-python-tuple-with-multiple-init-arguments"""
    def __add__(self, other):
        return self.construct(s+o for s,o in zip(self,other))
    def __call__(self, *args, **kwargs):
        return self.construct(x(*args, **kwargs) for x in self)


class Tuple(_Tuple,tuple):
    def construct(self, args):
        """Construct a new instance of type(self).
        args:
            A (Python)tuple of the things to construct with."""
        return type(self)(args)
    def __repr__(self):
        return '(x ' + ','.join(repr(x) for x in self) + ')'

class NamedTupleBase(_Tuple,tuple):
    """For use only with NamedTuple.  Accomodates different constructors
    for tuple vs namedtuple."""
    def construct(self, args):
        """Construct a new instance of type(self).
        args:
            A (Python)tuple of the things to construct with."""
        return type(self)(*args)

def NamedTuple(*args, **kwargs):
    return gicollections.namedtuple(*args, tuple_class=NamedTupleBase, **kwargs)
# -----------------------------------------------------------------

#class TupleFunction(_tuple, Function):
#    """If f,g are functions, creates a function H(*) <- (f(*), g(*))"""
#    def __init__(self, *funcs):
#        self.funcs = funcs
#    def __call__(self, *args, **kwargs):
#        return tuplex(fn(*args, *kwargs) for fn in self.funcs)
#    def __getitem__(self, ix):
#        return TupleWrap(self.value[ix])
#    def __repr__(self):
#        return '(*' + ','.join(repr(x) for x in self.funcs) + '*)'
# -------------------------------------------------------

#class TupleWrap(Function):
#    def __init__(self, value):
#        self.value = value
#    def __call__(self):
#        return tuplex(x() for x in self.value)
#    def __getitem__(self, ix):
#        return self.value[ix]
#    def __repr__(self):
#        return '(*' + ','.join(repr(x) for x in self.value) + '*)'
