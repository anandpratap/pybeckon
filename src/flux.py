import numpy as np
SMALL = 1e-15
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

    def speed(self, a, b):
        if abs(a-b) > SMALL:
            return (self.func(a) - self.func(b))/(a - b)
        else:
            return 0.0

    def __call__(self, u):
        return self.func(u)

class Burgers(Flux):
    def __init__(self):
        self.__name__ = "burgers"

    def func(self, u):
        return u*u/2.0

    def cspeed(self, u):
        return u

class Advection(Flux):
    
    def __init__(self, a = 1.0):
        self.a = a
        self.__name__ = "advection"
    def func(self, u):
        return self.a*u
    
    def cspeed(self, u):
        if type(u) == float:
            return self.a
        else:
            return self.a*np.ones_like(u)
    
    def actual(self, x, u_init, tf):
        u = np.zeros_like(x)
        dx = x[1] - x[0]
        deltaN = int(self.a*tf/dx)
        for i in range(deltaN, len(x)):
            u[i] = u_init[i-deltaN-1]
        return u


