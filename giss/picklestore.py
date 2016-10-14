import collections.abc
import checksum

class Picklestore(collections.abc.MutableMapping):

    def __init__(self, keystore, valuestore, key_fn=checksum.checksum):
        """bytestore:
            dict-like thing we rely upon in which keys and vals must be bytes"""
        self.keystore = keystore
        self.valuestore = valuestore
        self.key_fn = key_fn

    def __getitem__(self, key):
        tkey = self.key_fn(key)
        return self.bytestore.__getitem__(tkey)[1]

    def __setitem__(self, key):
