import types

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
    """Wraps a function in a class so we can add higher-order methods to it."""
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args, **argv):
        return self.fn(*args, **argv)

class BasicFunction(Function):
    """Wraps a raw Python function."""
    pass

xx = BasicFunction(arg_decorator)
print(dir(BasicFunction))
print(dir(xx))

# Method names to NOT lift when we mix

dontmix = set(('__call__', '__class__', '__delattr__', '__dict__',
              '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
              '__getattribute__', '__gt__', '__hash__', '__init__', '__le__',
              '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
              '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
              '__subclasshook__', '__weakref__'))


def mixin(obj, *klasses):
    """Mixes in all methods of klass into the instance obj.
    klasses:
        Classes or instances

    Another approach woudld be to dynamically change inheritance
    hierarchy of the class.  See:
    http://stackoverflow.com/questions/9539052/how-to-dynamically-change-base-class-of-instances-at-runtime?noredirect=1

    Especially:
        You can define a class object
            class Object(object):
                pass

        Which derives a class from the built-in metaclass type. That's
        it, now your new style classes can modify the __bases__
        without any problem.

        In my tests this actually worked very well as all existing
        (before changing the inheritance) instances of it and its
        derived classes felt the effect of the change including their
        mro getting updated.
    """
    for klass in klasses:
        print('Mixing in klass', klass)
        for name in dir(klass):
            if name in dontmix:
                continue
            method = getattr(klass, name)
            print('    {} {}'.format(type(method), method))
            print('    ', dir(method))
            if not isinstance(method, types.FunctionType):
                continue

            # https://filippo.io/instance-monkey-patching-in-python/
            setattr(obj, name, types.MethodType(method, obj))


def addops(*opclasses):
    """Decorator: Adds sets of higher-level operations to a function.
    opclasses:
        List of mixin classes containing the higher-level ops."""
    def real_decorator(fn):
        # Wrap the function as a class if needed
        if not isinstance(fn, Function):
            fn = BasicFunction(fn)

        mixin(fn, *opclasses)
        return fn

    return real_decorator

