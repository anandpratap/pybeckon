import numpy as np

class InitialFunction:
    def __init__(self):
        pass

    def init(self, x):
        pass
    
    def __call__(self, x):
        return self.init(x)

class Step(InitialFunction):
    def init(self, x):
        N = len(x)
        half = int(N/2)
        u = np.zeros_like(x)
        u[:N] = 1.0
        return u


class Square(InitialFunction):
    def init(self, x):
        N = len(x)
        quarter = int(N/4)
        u = np.zeros_like(x)
        u[quarter:3*quarter] = 1.0
        return u
        
