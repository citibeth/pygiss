import types
from giss import giutil

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
    def __call__(self, *args, **argv):
        return self.fn(*args, **argv)

class BasicFunction(Function):
    """Wraps a raw Python function."""
    pass

# Method names to NOT lift when we mix

dontmix = set(('__call__', '__class__', '__delattr__', '__dict__',
              '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
              '__getattribute__', '__gt__', '__hash__', '__init__', '__le__',
              '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
              '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
              '__subclasshook__', '__weakref__'))



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

