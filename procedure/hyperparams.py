
import itertools
import random

class ConstantParam:
    def __init__(self, x):
        self.x = x

    def __call__(self, **args):
        return self.x

class ChoiceParam:
    def __init__(self, *xs, mutation_rate=1.):
        self.xs = xs
        self.mutation_rate = mutation_rate

    def __call__(self, **args):
        if 'value' not in args or random.random() < self.mutation_rate:
            return random.choice(self.xs)
        else:
            return args['value']

class RangeParam:
    def __init__(self, lo=0, hi=1, mutation_rate=0., f=lambda x: x):  
        self.lo = lo
        self.hi = hi
        self.f = f
        self.mutation_rate = mutation_rate

    def __call__(self, value=None, **args):
        r = random.random()

        lo, hi = self.lo, self.hi
        
        if value is None or random.random() < self.mutation_rate:
            r = lo + (hi - lo) * r
            return self.f(r)
        else:
            return value * (0.8 + 0.4 * r)
    

