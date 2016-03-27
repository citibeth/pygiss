import pprint as xpprint

def to_dict(obj):
    """Reads all properties out of an f90wrap object."""
    ret = {}
    for attr,prop in type(obj).__dict__.items():
        if type(prop) == property:
            ret[attr] = prop.fget(obj)
    return ret

def pprint(obj):
    return xpprint.pprint(to_dict(obj))
