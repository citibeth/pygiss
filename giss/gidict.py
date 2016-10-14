import collections.abc
from giss import checksum
import pickle
import bsddb3.db

class Picklestore(collections.abc.MutableMapping):
    """Takes bytestring keys, stores Python objects"""

    def __init__(self, bytestore, prefix=b''):
        """bytestore:
            dict-like thing we rely upon in which keys and vals must be bytes"""
        self.bytestore = bytestore
        self.prefix = prefix

    def __getitem__(self, key):
        tkey = self.prefix + key
        return pickle.loads(self.bytestore[tkey])

    def __setitem__(self, key, value):
        tkey = self.prefix + key
        self.bytestore[tkey] = pickle.dumps(value)

    def __delitem__(self, key):
        tkey = self.prefix + key
        del self.bytestore[tkey]

    def __iter__(self):
        for tkey in iter(self.bytestore):
            if tkey.startswith(self.prefix):
                key = tkey[len(self.prefix):]
                yield key

    def __len__(self):
        return sum(1 for _ in iter(self))

    def close(self):
        self.bytestore.close()

class TransformDict(collections.abc.MutableMapping):
    def __init__(self, keystore, valuestore, key_fn=lambda x: x):
        self.keystore = keystore
        self.valuestore = valuestore
        self.key_fn = key_fn

    def __getitem__(self, key):
        tkey = self.key_fn(key)
        return self.valuestore[tkey]

    def __setitem__(self, key, value):
        tkey = self.key_fn(key)
        self.valuestore[tkey] = value

    def __delitem__(self, key):
        tkey = self.key_fn(key)
        del self.valuestore[tkey]
        del self.keystore[tkey]

    def __iter__(self):
        return iter(self.bytestore)

    def __len__(self):
        return len(self.bytestore)

    def close(self):
        self.keystore.close()
        self.valuestore.close()

def bdbdict(filename, dbtype=bsddb3.db.DB_HASH, flags=bsddb3.db.DB_CREATE, **kwargs):
    bdb = bsddb3.db.DB()
    bdb.open(filename, dbtype=dbtype, flags=flags, **kwargs)
    keystore = Picklestore(bdb, prefix=b'key_')
    valuestore = Picklestore(bdb, prefix=b'val_')
    return TransformDict(keystore, valuestore, key_fn=checksum.checksum)

