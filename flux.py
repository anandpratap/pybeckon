import numpy as np

class Flux:
    """
    Base class for flux function
    =============================

    Value of any flux can be calculatd using val function.
    
    function<func> contains the flux function
    function<speed> returns the wave velocity


    """
    
    def __init__(self):
        pass
    
    def func(self, u):
        pass

    def val(self, u):
        return self.func(u)

    def speed(self, u):
        return 1

    def __call__(self, u):
        return self.func(u)

class Burgers(Flux):

    def func(self, u):
        return u*u/2.0

    def speed(self, u):
            return u




class Advection(Flux):
    
    def __init__(self, a = 1.0):
        self.a = a
    
    def func(self, u):
        return self.a*u
    
    def speed(self, u):
        if type(u) is float:
            return self.a
        else:
            return self.a*np.ones_like(u)

