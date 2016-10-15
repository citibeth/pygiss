import types
from giss import giutil
from giss import functional
from giss.checksum import hashup
from giss import giutil
import types
import inspect

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

class Function(object):
    pass

# ---------------------------------------------------------
class Thunk(Function):
    """Base class for classes that acts like a function that returns a function."""
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
    def __call__(self):
        return self.fn(*self.args, **self.kwargs)
    hash_version=0
    def hashup(self,hash):
        hashup(hash, (self.fn, self.args, self.kwargs))

class thunkify(object):
    """Decorator that changes a function into a function that returns a
    thunk; that when called will run the original function.
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return Thunk(self.fn, *args, **kwargs)
# ---------------------------------------------------------
class BasicFunction(Function):
    """Wraps a function into a class so we can add higher-order methods to it."""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **argv):
        return self.fn(*args, **argv)


def addops(*opclasses):
    """Decorator: Wraps a Python function in a Function, and adds a set of higer-order
    operations to it.

    opclasses:
        List of mixin classes containing the higher-level ops."""
    def real_decorator(fn):
        # Wrap the function as a class if needed
        if isinstance(fn, Function):
            # ---- Give us a new class
            # http://stackoverflow.com/questions/8544983/dynamically-mixin-a-base-class-to-an-instance-in-python
            parents = giutil.uniq(list(fn.__bases__) + list(opclasses))
            name = '<{}>'.format(','.join(x.__name__ for x in parents))
            fn.__class__ = types.new_class(name, tuple(parents), {})
            return fn
        else:
            # ------ Create a new base class and instantiate it.
            # BasicFunction takes precedence over mixins
            parents = giutil.uniq([BasicFunction] + list(opclasses))
            name = '<{}>'.format(','.join(x.__name__ for x in parents))
            return types.new_class(name, tuple(parents), {})(fn)

    return real_decorator

# =================================================
# --------------------------------------------------------
class _arg(object):
    """Tagging class"""
    def __init__(self, index):
        self.index = index


class _Bind(functional.Function):
    """Reorder positional arguments.
    Eg: g = f('yp', _1, 17, _0, dp=23)
    Then g('a', 'b', another=55) --> f('yp', 'b', 17, 'a', dp=23, another=55)

    TODO:
       1. When wrapping multiple _Binds, create just a single level???
       2. Get kwargs working
    """

    def __init__(self, fn, *pargs, **pkwargs):
        # Maximum index referred to by the user.
        # Inputs to f above this index will be passed through
        self.fn = fn.fn if isinstance(fn, functional.BasicFunction) else fn    # Avoid unnecessary wrapping
        self.pargs = pargs
        self.pkwargs = pkwargs
        self.max_gindex = max(
            (x.index if isinstance(x, _arg) else -1 for x in pargs),
            default=-1)

    def __call__(self, *gargs, **gkwargs):
        fargs = \
            [gargs[x.index] if isinstance(x, _arg) else x for x in self.pargs] + \
            list(gargs[self.max_gindex+1:])

        fkwargs = dict(self.pkwargs)
        fkwargs.update(gkwargs)    # Overwrite keys
        print('*** fargs', self.fn, fargs)
        return self.fn(*fargs, **fkwargs)

def bind(fn, *pargs, **pkwargs):
    # Lift ops to the bound function
    if isinstance(fn, functional.Function):
        parents = giutil.uniq([_Bind] + list(type(fn).__bases__))
        name = '<{}>'.format(','.join(x.__name__ for x in parents))
        klass = types.new_class(name, tuple(parents), {})
    else:
        klass = _Bind
        # No metadata to copy on a raw function

    return klass(fn, *pargs, **pkwargs)

