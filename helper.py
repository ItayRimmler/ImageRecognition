from exceptions import tooSmall
import random as r

def batch(x, y, size, DATA_SET_SIZE):
    try:

        size = round(size)

        if size > DATA_SET_SIZE or size < 1:
            raise tooSmall(DATA_SET_SIZE, size)

    except tooSmall as e:
        size = e.b

    ind = sorted(r.sample(range(0, DATA_SET_SIZE), size))
    x = x[ind]
    y = y[ind]
    return x, y