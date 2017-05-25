from math import log
from functools import reduce

def shannon(vec,**kwargs):
    reduce(lambda y, z: y + z * log(z, kwargs.get("shanon_base", 2)),
           vec.value_counts().map(lambda x: x / vec.shape[0]))

def dshannon(data,xlabel,ylabel,**kwargs):
           data[xlabel].value_counts().index.map(lambda x: shannon(data[ylabel][data[xlabel] == x]),**kwargs)

