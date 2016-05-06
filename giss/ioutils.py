
# http://stackoverflow.com/questions/13250050/redirecting-the-output-of-a-python-function-from-stdout-to-variable-in-python
@contextmanager
def redirect_io(out=sys.stdout, err=sys.stderr):
    saved = (sys.stdout, sys.stderr)
    sys.stdout = out
    sys.stderr = err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved
