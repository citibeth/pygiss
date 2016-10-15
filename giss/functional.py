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

# http://stackoverflow.com/questions/9539052/how-to-dynamically-change-base-class-of-instances-at-runtime?noredirect=1
class Object(object):
    pass

class Function(Object):
    """Wraps a function in a class so we can add higher-order methods to it."""
    def __init__(self, fn):
        self.fn = fn
        self.attrs = dict()    # Meta-data

    def __call__(self, *args, **argv):
        return self.fn(*args, **argv)


class BasicFunction(Function):
    """Wraps a raw Python function."""
    pass


def addops(*opclasses):
    """Decorator: Adds sets of higher-level operations to a function.
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


#class Thunk(functional.Function):
#    """Simple version of bind"""
#    def __init__(self, func, *args, **kwargs):
#        self.func = func
#        self.args = args
#        self.kwargs = kwargs
#    def __call__(self):
#        return self.func(*self.args, **self.kwargs)
#
#    hash_version = 0
#    def hashup(self, hash):
#        hashup(hash, self.func)
#        hashup(hash, self.args)
#        hashup(hash, self.kwargs)

class _Bind(functional.Function):
    """Reorder positional arguments.
    Eg: g = f('yp', _1, 17, _0, dp=23)
    Then g('a', 'b', another=55) --> f('yp', 'b', 17, 'a', dp=23, another=55)

    TODO:
       1. When wrapping multiple _Binds, create just a single level???
       2. Get kwargs working
    """

    def __init__(self, fn, attrs, *pargs, **pkwargs):
        # Maximum index referred to by the user.
        # Inputs to f above this index will be passed through
        self.fn = fn.fn if isinstance(fn, functional.BasicFunction) else fn    # Avoid unnecessary wrapping
        self.pargs = pargs
        self.pkwargs = pkwargs
        self.max_gindex = max(
            (x.index if isinstance(x, _arg) else -1 for x in pargs),
            default=-1)

        # Figure out this function's signature
        # ...and put bound parameters into the metadata
        if isinstance(fn, functional.Function):
            sig = fn.signature
        else:
            sig = inspect.signature(self.fn)
        bound = sig.bind_partial(*pargs, **pkwargs).arguments
        old_pl = list(sig.parameters.values())
        new_pl = [None]*(self.max_gindex+1)
        for param in old_pl:
            if param.name in bound:
                val = bound[param.name]
                if isinstance(val, _arg):
                    # Not really bound here; put it in output param list
                    new_pl[val.index] = param
                else:
                    # Parameter was bound here; add it to the metadata
                    attrs[(fn.qualname,param.name)] = bound[param.name]
            else:
                # Parameter not bound; hopefully it's a kwarg or a tailing
                # positional arg.  Either way, append to parameter list
                new_pl.append(param)
        new_sig = sig.replace(parameters=new_pl)
        self.signature = new_sig
        self.attrs = attrs

    def __call__(self, *gargs, **gkwargs):
        fargs = \
            [gargs[x.index] if isinstance(x, _arg) else x for x in self.pargs] + \
            list(gargs[self.max_gindex+1:])

        fkwargs = dict(self.pkwargs)
        fkwargs.update(gkwargs)    # Overwrite keys
        print('fargs', fargs)
        return self.fn(*fargs, **fkwargs)

def bind(fn, *args, attrs=None, **kwargs):
    if attrs is None:
        attrs = dict()

    # Lift ops to the bound function
    if isinstance(fn, functional.Function):
        parents = giutil.uniq([_Bind] + list(type(fn).__bases__))
        name = '<{}>'.format(','.join(x.__name__ for x in parents))
        klass = types.new_class(name, tuple(parents), {})
        attrs.extend(fn.attrs)   # Start with metadata from bound function...
    else:
        klass = _Bind
        # No metadata to copy on a raw function

    return klass(fn, attrs, *pargs, **pkwargs)

