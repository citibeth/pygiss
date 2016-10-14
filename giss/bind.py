from giss import functional
from giss.checksum import hashup

# See: functoolspartial for binding...


class _arg(object):

    """Tagging class"""
    def __init__(self, index):
        self.index = index


class Thunk(functional.Function):
    """Simple version of bind"""
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def __call__(self):
        return self.func(*self.args, **self.kwargs)

    hash_version = 0
    def hashup(self, hash):
        hashup(hash, self.func)
        hashup(hash, self.args)
        hashup(hash, self.kwargs)

class bind(functional.Function):
    """Reorder positional arguments.
    Eg: g = f('yp', _1, 17, _0, dp=23)
    Then g('a', 'b', another=55) --> f('yp', 'b', 17, 'a', dp=23, another=55)
    """

    def __init__(self, fn, *pargs, **pkwargs):
        # Maximum index referred to by the user.
        # Inputs to f above this index will be passed through

        # Lift the core function to avoid unnecessary wrapping
        if isinstance(fn, functional.BasicFunction):
            self.fn = fn.fn
        else:
            self.fn = fn

        self.pargs = pargs
        self.pkwargs = pkwargs
        self.max_gindex = max(
            (x.index if isinstance(x, _arg) else -1 for x in pargs),
            default=-1)

        # Lift ops from function we wrapped
        if isinstance(fn, functional.Function):
            functional.mixin(self.fn, (fn,))

    def __call__(self, *gargs, **gkwargs):
        fargs = \
            [gargs[x.index] if isinstance(x, _arg) else x for x in self.pargs] + \
            list(gargs[self.max_gindex+1:])

        fkwargs = dict(self.pkwargs)
        fkwargs.update(gkwargs)    # Overwrite keys
        return self.fn(*fargs, *fkwargs)
