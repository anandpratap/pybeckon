import numpy as np

class InitialFunction:
    def __init__(self):
        pass

    def init(self, x):
        pass
    
    def __call__(self, x):
        return self.init(x, t=0.0)

class Step(InitialFunction):
    def init(self, x, t):
        N = len(x)
        half = int(N/2)
        u = np.zeros_like(x)
        u[:N] = 1.0
        return u

class InvertedStep(InitialFunction):
    def init(self, x):
        N = len(x)
        half = int(N/2)
        u = np.zeros_like(x)
        u[:half] = 0.4
        u[half:] = 1.2
        return u


class Square(InitialFunction):
    def init(self, x, t):
        N = len(x)
        quarter = int(N/8)
        u = np.zeros_like(x)
        u[3*quarter:5*quarter] = 1.0
        return u

class Sin(InitialFunction):
    def init(self, x, t):
        return np.sin(np.pi*(x - t))

        
class Complex(InitialFunction):
    def init(self, x, t):
        x_min = min(x)
        x_max = max(x)
        L = x_max - x_min
        x_ = np.zeros(8)
        u = np.zeros_like(x)
        N = len(x)
        x_[0] = -1.0 + t
        x_[1] = -0.75 + t
        x_[2] = -0.5 + t
        x_[3] = 0.0 + t
        x_[4] = 0.25 + t
        x_[5] = 0.75 + t
        x_[6] = 1.0 + t
        x_[7] = 1.25 + t
        for i in range(N):
            if x[i] >= x_[0] and x[i] < x_[1]:
                u[i] = (x[i] - x_[0])*4
            elif x[i] >= x_[2] and x[i] < x_[3]:
                u[i] = (x[i] - x_[2])*(x[i] - x_[3])*15
            elif x[i] >= x_[4] and x[i] < x_[5]:
                u[i] = 1.2
            elif x[i] >= x_[6] and x[i] < x_[7]:
                u[i] = -1.0
        return u
