from pylab import *

import random as rand

from flux import Burgers, Advection
from utils import NewtonDD
from initial import Step, Square
from timestepping import Euler, RK2

SMALL = 1e-12




class Solver:

    def __init__(self, xlim=2, N=200, CFL=0.8, Nb=10, order = 3):
        self.x = linspace(-xlim , xlim, N+1)
        self.N = N
        self.Nb = Nb
        self.CFL = CFL
        self.xc = zeros(self.N)
        self.u = zeros(self.N)
        self.f = Burgers()
        self.order = order
        
    def set_centroid(self):
        for i in range(self.N):
            self.xc[i] = 0.5*(self.x[i] + self.x[i+1])
        self.dx = self.xc[1] - self.xc[0]

    def set_initial(self, ifunc = Square()):
        self.u = copy(ifunc(self.xc))
        print self.u
        
    def setup(self, ifunc = Square()):
        self.set_centroid()
        self.set_initial(ifunc = ifunc)

    def calc_shift(self, i, order):
        if abs(self.u[i+1]-self.u[i]) > SMALL:
            abar = (self.f(self.u[i+1]) - self.f(self.u[i]))/(self.u[i+1]-self.u[i])
        else:
            abar = 0
        if abar >= 0:
            k = i
        else:
            k = i + 1
        L = 2
        for i in range(order-1):
            a = self.ndd.calc(k, k+L)
            b = self.ndd.calc(k-1, k+L-1)
            if abs(a) >= abs(b):
                k = k-1
            L +=  1

        return k

    def calc_val(self, i):
        self.ndd = NewtonDD(self.x, self.u, self.f)
        L = self.calc_shift(i, 3)
        r = i - L
        if r == -1:
            v = 11.0/6.0*self.u[i-r+0] - 7.0/6.0*self.u[i-r+1] + 1.0/3.0*self.u[i-r+2]
        elif r == 0:
            v = 1.0/3.0*self.u[i-r+0] + 5.0/6.0*self.u[i-r+1] - 1.0/6.0*self.u[i-r+2]
        elif r == 1:
            v = -1.0/6.0*self.u[i-r+0] + 5.0/6.0*self.u[i-r+1] + 1.0/3.0*self.u[i-r+2]
        elif r == 2:
            v = 1.0/3.0*self.u[i-r+0] - 7.0/6.0*self.u[i-r+1] + 11.0/6.0*self.u[i-r+2]
        else:
            print "Error"
        return v

    def calc_residue(self):
        v = np.zeros(self.N)
        flux = np.zeros(self.N)
        res = np.zeros(self.N)
        self.alpha = max(abs(self.f.speed(self.u)))
        for i in range(self.order, self.N - self.order):
            v[i] = self.calc_val(i)
        for i in range(self.order+1, self.N - self.order-1):
            flux[i] = 0.5*(self.f(v[i]) + self.f(v[i+1]) - self.alpha*(v[i+1]-v[i]))
        for i in range(self.order+2, self.N - self.order-2):
            dx = self.x[i+1]-self.x[i]
            res[i] = -(flux[i] - flux[i-1])/dx
        return res
    

    def calc_dt(self):
        umax = max(abs(self.u))
        return self.CFL*(self.x[1]-self.x[0])/umax


s = Solver()
s.setup()
plot(s.xc, s.u, 'x-')
s.CFL = 0.4
I = RK2(solver=s, tf=0.66)
I.run()
plot(s.xc, s.u, 'x-')

# s = Solver()
# s.setup()
# s.CFL = 0.4
# I = Euler(solver=s, tf=0.66)
# I.run()
# plot(s.xc, s.u, 'x-')
# # s.setup()
# legend(['Initial','RK2','Euler'])

# s = Solver()
# s.setup()
# s.cfl = 0.4
# s.solve(0.66, 2)
# plot(s.xc, s.u, 'x-')

# s = Solver()
# s.setup()
# s.cfl = 0.4
# s.solve(0.66, 3)
# plot(s.xc, s.u, 'x-')

# s = Solver()
# s.setup()
# s.cfl = 0.1
# s.solve(0.66, 4)



show()
