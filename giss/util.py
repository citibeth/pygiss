class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

# Prints a summary of a numpy array
def numpy_stype(var) :
	return ''.join([str(var.dtype), str(var.shape)])
