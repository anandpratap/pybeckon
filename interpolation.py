from pylab import *

import random as rand
from flux import Burgers, Advection
from utils import NewtonDD

SMALL = 1e-19




class Solver:
    def __init__(self):
        self.N = 100
        self.x = linspace(-2,2,self.N+1)
        self.Nb = 10
        self.xc = zeros(self.N)
        self.u = zeros(self.N)
        self.cfl = 0.4
        for i in range(self.N):
            self.xc[i] = 0.5*(self.x[i] + self.x[i+1])
            if abs(self.xc[i]) < 1/3.0:
                self.u[i] = 1
            
        self.dx = self.xc[1] - self.xc[0]
        self.f = Burgers()
        self.order = 3

        
    def calc_poly(self, i, order):
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

    def step(self):
        u_old = copy(self.u)
        u_new = copy(self.u)
        self.ndd = NewtonDD(self.x, u_old, self.f)
        f = zeros_like(self.u)
        for i in range(self.Nb, self.N-self.Nb):
            L = self.calc_poly(i)
            d = zeros([self.order+1,2])
            d[0][0] = 1
            d[0][1] = 1
            sum = 0
            for k in range(self.order):
                sum += d[k][1]*self.ndd.calc(L, L+k+1)
                d[k+1][0] = (self.x[i] - self.x[L+k-1])*d[k][0]
                d[k+1][1] = d[k+1][0] + (self.x[i]-self.x[L+k])*d[k][1]
            f[i] = sum
            
        for i in range(self.Nb+1, self.N-self.Nb-1):
            self.u[i] -= self.dt/self.dx*(f[i]-f[i-1])

    def newstep(self):
        u_old = copy(self.u)
        u_new = copy(self.u)
        self.ndd = NewtonDD(self.x, u_old, self.f)
        f = zeros_like(self.u)
        v = zeros_like(self.u)
        for i in range(self.Nb+4, self.N-self.Nb-4):
            L = self.calc_poly(i, 3)
            r = i - L
            if r == -1:
                v[i] = 11.0/6.0*u_old[i-r+0] - 7.0/6.0*u_old[i-r+1] + 1.0/3.0*u_old[i-r+2]
            elif r == 0:
                v[i] = 1.0/3.0*u_old[i-r+0] + 5.0/6.0*u_old[i-r+1] - 1.0/6.0*u_old[i-r+2]
            elif r == 1:
                v[i] = -1.0/6.0*u_old[i-r+0] + 5.0/6.0*u_old[i-r+1] + 1.0/3.0*u_old[i-r+2]
            elif r == 2:
                v[i] = 1.0/3.0*u_old[i-r+0] - 7.0/6.0*u_old[i-r+1] + 11.0/6.0*u_old[i-r+2]
            else:
                print "Error"
        for i in range(self.Nb+5, self.N-self.Nb-5):
            self.alpha = 1.01
            #f[i] = 0.5*(self.f(v[i]) + self.f(v[i+1]) - self.alpha*(v[i+1]-v[i]))
            f[i] = self.f(v[i])
        for i in range(self.Nb+6, self.N-self.Nb-6):
            self.u[i] -= self.dt/self.dx*(f[i]-f[i-1])

    def step4(self):
        u_old = copy(self.u)
        u_new = copy(self.u)
        self.ndd = NewtonDD(self.x, u_old, self.f)
        f = zeros_like(self.u)
        v = zeros_like(self.u)
        for i in range(self.Nb+5, self.N-self.Nb-5):
            L = self.calc_poly(i, 4)
            r = i - L
            if r == -1:
                v[i] = 25.0/12.0*u_old[i-r+0] - 23.0/12.0*u_old[i-r+1] + 13.0/12.0*u_old[i-r+2] - 1.0/4.0*u_old[i-r+3]
            elif r == 0:
                v[i] = 1.0/4.0*u_old[i-r+0] + 13.0/12.0*u_old[i-r+1] - 5.0/12.0*u_old[i-r+2] + 1.0/12.0*u_old[i-r+3]
            elif r == 1:
                v[i] = -1.0/12.0*u_old[i-r+0] + 7.0/12.0*u_old[i-r+1] + 7.0/12.0*u_old[i-r+2] - 1.0/12.0*u_old[i-r+3]
            elif r == 2:
                v[i] = 1.0/12.0*u_old[i-r+0] - 5.0/12.0*u_old[i-r+1] + 13.0/12.0*u_old[i-r+2] + 1.0/4.0*u_old[i-r+3]
            elif r == 3:
                v[i] = -1.0/4.0*u_old[i-r+0] + 13.0/12.0*u_old[i-r+1] - 23.0/12.0*u_old[i-r+2] + 25.0/12.0*u_old[i-r+3]
            else:
                print "Error"
        for i in range(self.Nb+6, self.N-self.Nb-6):
            self.alpha = 1.01
            #f[i] = 0.5*(self.f(v[i]) + self.f(v[i+1]) - self.alpha*(v[i+1]-v[i]))
            f[i] = self.f(v[i])
        for i in range(self.Nb+7, self.N-self.Nb-7):
            self.u[i] -= self.dt/self.dx*(f[i]-f[i-1])

    def step2(self):
        u_old = copy(self.u)
        u_new = copy(self.u)
        self.ndd = NewtonDD(self.x, u_old, self.f)
        f = zeros_like(self.u)
        v = zeros_like(self.u)
        for i in range(self.Nb+5, self.N-self.Nb-5):
            L = self.calc_poly(i, 2)
            r = i - L
            if r == -1:
                v[i] = 3.0/2.0*u_old[i-r+0] - 1.0/2.0*u_old[i-r+1]
            elif r == 0:
                v[i] = 1.0/2.0*u_old[i-r+0] + 1.0/2.0*u_old[i-r+1] 
            elif r == 1:
                v[i] = -1.0/2.0*u_old[i-r+0] + 3.0/2.0*u_old[i-r+1] 
            else:
                print "Error"
        for i in range(self.Nb+6, self.N-self.Nb-6):
            self.alpha = 1.01
            #f[i] = 0.5*(self.f(v[i]) + self.f(v[i+1]) - self.alpha*(v[i+1]-v[i]))
            f[i] = self.f(v[i])
        for i in range(self.Nb+7, self.N-self.Nb-7):
            self.u[i] -= self.dt/self.dx*(f[i]-f[i-1])


    def set_dt(self):
        umax = max(abs(self.u))
        self.dt = self.cfl*(self.x[1]-self.x[0])/umax
        print self.dt

    def analytical(self, tf):
        shift = tf/self.dx
        u = copy(self.u)
        for i in range(self.N):
            try:
                u[i+shift] = self.u[i]
            except:
                pass
        plot(self.xc, u, 'x-')
        
    def solve(self, tf, order):
        t = 0
        while t < tf:
            self.set_dt()
            if order == 3:
                self.newstep()
            elif order == 4:
                self.step4()
            elif order == 2:
                self.step2()
            else:
                break
            t += self.dt

s = Solver()
plot(s.xc, s.u, 'x-')

s = Solver()
s.cfl = 0.4
s.solve(0.66, 2)
plot(s.xc, s.u, 'x-')

s = Solver()
s.cfl = 0.4
s.solve(0.66, 3)
plot(s.xc, s.u, 'x-')

s = Solver()
s.cfl = 0.1
s.solve(0.66, 4)
plot(s.xc, s.u, 'x-')

legend(['Exact','ENO2','ENO3 cfl=0.4', 'ENO4 cfl=0.2'])
show()
