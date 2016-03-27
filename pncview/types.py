from giss import giutil

class VarEntry(giutil.SlotStruct):
    __slots__ = ('type', 'vname', 'shape', 'formula')
