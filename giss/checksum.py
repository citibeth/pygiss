import hashlib
import struct
import types

# Our chosen hash function
begin = hashlib.md5

"""To make a class checksummable:

   1. Add tag hash_version, which you will change when you want to
      change the hash.
   2. Add as hashup() method.

   Eg:
   class MyClass(object):
       hash_version = 17
       def __init__(self):
           self.val = 17
       def hashup(self, hash):
           hashup(hash, self.val)
"""
# --------------------------------------------------------
class CheckFile(object):
    """Used to mark a path whose modification date and size should be
    included in the hash."""

    def __init__(self, fname):
        self.fname = fname

    hash_version=0
    def hashup(hash):
        hashup(hash, self.fname)
        hashup(hash, os.path.getmtime(path))
        hashup(hash, os.path.getsize(path))

# --------------------------------------------------------
def hashup_int(hash, x):
    try:
        # https://docs.python.org/2/library/struct.html#format-characters
        bytes = struct.pack('>i',x)
    except struct.error:    # x is too large
        bytes = str(n).encode()
    hash.update(bytes)

def hashup_float(hash, x):
    hash.update(struct.pack('>f',x))

def hashup_str(hash, x):
    hash.update(x.encode())

def hashup_bytes(hash, x):
    hash.update(x)

def hashup_sequence(hash, coll):
    hash.update(b'sequence')
    for x in coll:
        hashup(hash, x)

def hashup_set(hash, coll):
    hashup_sequence(sorted(tuple(coll)))

def hashup_dict(hash, coll):
    hash.update(b'dict')
    hashup_sequence(hash, sorted(tuple(coll.items())))

def hashup_fn(hash, fn):
    hash.update(b'function')
    hash.update(fn.__module__.encode())
    hash.update(fn.__qualname__.encode())  # https://www.python.org/dev/peps/pep-3155/

def hashup_method(hash, method):
    hash.update(b'method')
    hashup(hash, method.__self__)
    hash.update(method.__module__.encode())
    hash.update(method.__qualname__.encode())  # https://www.python.org/dev/peps/pep-3155/


def hashup_module(hash, mod):
    hash.update(b'module')
    hash.update(mod.__package__.encode())
    hash.update(mod.__name__.encode())

def hashup_type(hash, klass):
    hash.update(b'class')
    hash.update(klass.__module__.encode())
    hash.update(klass.__qualname__.encode())


# -----------------------------------------
def hashup_error(hash, x):
    raise ValueError('Cannot checksum {} {}'.format(type(x), x))

hashup_methods = {
    int : hashup_int,
    float : hashup_float,
    str : hashup_str,
    bytes : hashup_bytes,
    tuple : hashup_sequence,
    list : hashup_sequence,
    set : hashup_set,
    dict : hashup_dict,
    type : hashup_type,
    types.FunctionType : hashup_fn,
    types.MethodType : hashup_method,
    types.GeneratorType : hashup_fn,
    types.CoroutineType : hashup_fn,
    types.BuiltinFunctionType : hashup_fn,
    types.BuiltinMethodType : hashup_fn,
    types.ModuleType : hashup_module,
    pathutils.Path : hashup_path,
}

def hashup(hash, x, klass=None):
    # When hashing parameters we sometimes know what the class will be
    if klass is None:
        klass = type(x)
    if klass in hashup_methods:
        hashup_methods[klass](hash, x)
    else:
        hash.update(b'object')
        hash.update(klass.__module__.encode())
        hash.update(klass.__name__.encode())
        hashup(hash, klass.hash_version)
        x.hashup(hash)

def checksum(x):
    """Top-level function"""
    hash = begin()
    hashup(hash, x)
    return hash.digest()
