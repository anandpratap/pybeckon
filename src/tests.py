import numpy as np
from flux import Advection, Burgers
from timestepping import RK2, RK4, Euler
from initial import Step, InvertedStep, Complex, Square
from solver import Solver
from pylab import *

class Test:
    def __init__(self):
        pass

class CompareOrders(Test):
    def  __init__(self, tf = [0.1], orders = [1], dx = 0.01, xlim=1):
        self.tf = tf
        self.orders = orders
        self.dx = dx
        self.CFL = {1:0.8, 2:0.2, 3:0.2, 4:0.1}
        self.scheme = 2
        self.xlim = xlim 
        self.N = int(2*self.xlim/self.dx)
        self.equation = Advection()
        self.ifunc = Complex()
        self.shift = True
        self.legend = []

    def plotanalytical(self, t):
        s = Solver(f = self.equation, xlim=self.xlim, N=self.N, CFL=0.8, order=1, ifunc=self.ifunc, scheme=self.scheme)
        u_actual = self.ifunc.init(s.xc, t)
        plot(s.xc, u_actual, 'x-')
        self.legend.append("Analytical Time: %s" %(t))

    def test(self, t, order):
        s = Solver(f = self.equation, xlim=self.xlim, N=self.N, CFL=self.CFL[order], order=order, ifunc=self.ifunc, scheme=self.scheme)
        I = RK2(solver=s, tf=t, shift=self.shift)
        I.run()
        plot(s.xc, s.u, 'x-')
        self.legend.append("Order: %s, Time: %s"%(order, t))
        
    def run(self):
        for t in self.tf:
            figure()
            self.legend = []
            self.plotanalytical(t)
            for order in self.orders:
                self.test(t, order)
                print "Testing Complete......tf: %s, Order: %s "%(t, order)
            legend(self.legend)
            grid()
            savefig("./png/time%s.png"%(t))


    def multiplot(self, tf = [1.0]):
        for t in tf:
            figure()
            filename = "./data/%s_%s_%s_%s_%s.npz"%(self.equation.__name__, 1, self.scheme, self.N, t)
            data = np.load(filename)
            x = data['x']
            uactual = data['uactual']
            plot(x, uactual, 'x-', label='analytical')
            for order in self.orders:
                self.plot(t, order)
            legend()
            grid(True)
            show()

    def plot(self, t, order):
        filename = "./data/%s_%s_%s_%s_%s.npz"%(self.equation.__name__, order, self.scheme, self.N, t)
        data = np.load(filename)
        x = data['x']
        u = data['u']
        plot(x, u, 'x-', label='Order %s Time %s'%(order, t))
        
if __name__ == "__main__":
    tf = [1.0, 2.0, 3.0, 4.0, 5.0]
    orders = [1, 2, 3, 4]
    test = CompareOrders(orders=orders, tf=tf, xlim=15)
    test.run()
