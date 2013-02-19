import numpy as np
from flux import Advection, Burgers
from timestepping import RK2, RK4, Euler
from initial import Step, InvertedStep, Complex, Square, Sin
from solver import Solver
from pylab import *


class Test:
    def __init__(self):
        pass


    
class CompareOrders(Test):
    def  __init__(self, tf = [0.1], orders = [1], dx = 0.01, xlim=1, scheme=2, equation=Advection()):
        self.tf = tf
        self.orders = orders
        self.dx = dx
        self.CFL = {1:0.8, 2:0.6, 3:0.5, 4:0.4, 5:0.4, 7:0.4, 9:0.4}
        self.scheme = scheme
        self.xlim = xlim 
        self.N = int(2*self.xlim/self.dx)
        self.equation = equation
        self.ifunc = Complex()
        self.shift = True
        self.legend = []

    def plotanalytical(self, t):
        s = Solver(f = self.equation, xlim=self.xlim, N=self.N, CFL=0.8, order=1, ifunc=self.ifunc, scheme=self.scheme)
        u_actual = self.ifunc.init(s.xc, t)
        plot(s.xc, u_actual, 'x-')
        self.legend.append("Analytical Time: %s" %(t))

    def test(self, t, order):
        self.s = Solver(f = self.equation, xlim=self.xlim, N=self.N, CFL=self.CFL[order], order=order, ifunc=self.ifunc, scheme=self.scheme)
        I = RK2(solver=self.s, tf=t, shift=self.shift)
        I.run()
        #u_actual = self.ifunc.init(self.s.xc, t)
        #error = abs((u_actual - self.s.u))
        #print max(error)
        plot(self.s.xc, self.s.u, 'x-')
        self.legend.append("Order: %s, Time: %s"%(self.s.scheme_map[order].__name__, t))
        
    def run(self):
        for t in self.tf:
            figure()
            self.legend = []
            self.plotanalytical(t)
            for order in self.orders:
                self.test(t, order)
                print "Testing Complete......tf: %s, Order: %s %s"%(t, order, self.s.scheme_map[order].__name__)

            legend(self.legend)
            grid()
            savefig("./png/time%s.png"%(t))
            #show()

    def multiplot(self, tf = [1.0]):
        for t in tf:
            #figure()
            filename = "./data/%s_%s_%s_%s_%s.npz"%(self.equation.__name__, 1, self.scheme, self.N, t)
            data = np.load(filename)
            x = data['x']
            uactual = data['uactual']
            pyplot.plot(x, uactual, 'x-', label='analytical')
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
    orders = [1, 2, 3, 4, 5, 7, 9]
    equation = Advection()
    test = CompareOrders(xlim=10, tf=tf, orders=orders, scheme=2, equation=equation)
    test.run()
