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

