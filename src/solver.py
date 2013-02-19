from pylab import *

import random as rand

from flux import Burgers, Advection
from utils import NewtonDD
from initial import Step, Square, InvertedStep, Complex
from timestepping import Euler, RK2, CudaEuler

import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

SMALL = 1e-12



class Solver:
    """
    Class to setup and solve 1D wave equation
    ============================================
    
    Input
    ------
    f    : Flux object, available option are
           Advection (default)
           Burgers
           Custom fluxes can be defined as given in flux.py

    xlim : Domain is then set in [-xlim, xlim]
    N    : Number of cells
    CFL  : CFL number
    order: order of schemes, available orders are
           1 - First order upwind
           2 - Second order ENO
           3 - Third order ENO
           4 - Fourth order ENO
           5 - Fifth order ENO
           6 - Sixth order ENO
           7 - Seventh order ENO

    -------------------------------------------
    Example Usage
    -------------------------------------------

    s = Solver(f=Burgers(), xlim=1, N=100, CFL=0.4, order=3)
    # setup the solver
    s.setup()
    # define integrator

    I = RK2(solver=s, tf=1.0)
    I.run()


    """
    def __init__(self, f=Advection(), xlim=2, N=200, CFL=0.8, order = 3, ifunc=Square(), scheme = 1):
        self.x = linspace(-xlim , xlim, N+1)
        self.N = N
        self.CFL = CFL
        self.xc = zeros(self.N)
        self.u = zeros(self.N)
        self.f = f
        self.order = order
        self.scheme_map = {1:self.Godunov, 2:self.ENO2, 3:self.ENO3, 4:self.ENO4, 5:self.WENO5, 7:self.WENO7, 9:self.WENO9}
        self.set_centroid()
        self.set_initial(ifunc=ifunc)
        self.scheme = scheme
        
    def set_centroid(self):
        for i in range(self.N):
            self.xc[i] = 0.5*(self.x[i] + self.x[i+1])
        self.dx = self.xc[1] - self.xc[0]

    def set_initial(self, ifunc = Square()):
        self.u = copy(ifunc(self.xc))


    def calc_shift(self, i, start, order):
        if start == i - 1:
            r = 0
        else:
            r = -1
        lower = start
        upper = start
        for i in range(1, order):
            a = self.ndd.calcu(lower-1, upper)
            b = self.ndd.calcu(lower, upper+1)
            if abs(a) < abs(b):
                r += 1
                lower -= 1
            else:
                upper += 1
        return r


    def calc_val(self, i):
            return self.scheme_map[self.order](i)


    def Godunov(self, i):
        vl = self.u[i-1]
        vr = self.u[i]
        return [vl, vr]


    def WENO7(self, i):
        pass


    def ENO2(self, i):
        c = {-1:[3.0/2.0,-1.0/2.0],
              0:[1.0/2.0,1.0/2.0],
              1:[-1.0/2.0,3.0/2.0]
              }
        self.ndd = NewtonDD(self.x, self.u, self.f)
        
        r = self.calc_shift(i, i-1, 2)
        I = i-1
        vl = c[r][0]*self.u[I-r+0] + c[r][1]*self.u[I-r+1]
        
        
        r = self.calc_shift(i, i, 2)
        I = i-1
        vr = c[r][0]*self.u[I-r+0] + c[r][1]*self.u[I-r+1]
        
        return [vl, vr]


    def ENO3(self, i):
        c = {-1:[11.0/6.0,-7.0/6.0,1.0/3.0],
              0:[1.0/3.0,5.0/6.0,-1.0/6.0],
              1:[-1.0/6.0,5.0/6.0,1.0/3.0],
              2:[1.0/3.0,-7.0/6.0,11.0/6.0]
              } 
        self.ndd = NewtonDD(self.x, self.u, self.f)
        r = self.calc_shift(i, i-1, 3)
        I = i-1
        vl = c[r][0]*self.u[I-r+0] + c[r][1]*self.u[I-r+1] + c[r][2]*self.u[I-r+2] 
        
        
        r = self.calc_shift(i, i, 3)
        I = i-1
        vr = c[r][0]*self.u[I-r+0] + c[r][1]*self.u[I-r+1] + c[r][2]*self.u[I-r+2] 
        
        return [vl, vr]

    def ENO4(self, i):
        c = {-1:[25.0/12.0,-23.0/12.0,13.0/12.0,-1.0/4.0],
              0:[1.0/4.0,13.0/12.0,-5.0/12.0,1.0/12.0],
              1:[-1.0/12.0,7.0/12.0,7.0/12.0,-1.0/12.0],
              2:[1.0/12.0,-5.0/12.0, 13.0/12.0,1.0/4.0],
              3:[-1.0/4.0,13.0/12.0,-23.0/12.0,25.0/12.0],
              } 

        self.ndd = NewtonDD(self.x, self.u, self.f)
        r = self.calc_shift(i, i-1, 4)
        I = i-1
        vl = c[r][0]*self.u[I-r+0] + c[r][1]*self.u[I-r+1] + c[r][2]*self.u[I-r+2] + c[r][3]*self.u[I-r+3] 
        
        r = self.calc_shift(i, i, 4)
        I = i-1
        vr = c[r][0]*self.u[I-r+0] + c[r][1]*self.u[I-r+1] + c[r][2]*self.u[I-r+2] + c[r][3]*self.u[I-r+3] 
        
        return [vl, vr]

    def WENO5(self, i):
        vl = self.WENO5_p(i, 1)
        vr = self.WENO5_p(i, -1)
        return [vl, vr]

    def WENO7(self, i):
        vl = self.WENO7_p(i, 1)
        vr = self.WENO7_p(i, -1)
        return [vl, vr]

    def WENO9(self, i):
        vl = self.WENO9_p(i, 1)
        vr = self.WENO9_p(i, -1)
        return [vl, vr]


    def WENO5_p(self, i, wavespeed):
        start = i-1
	ustartm2 = self.u[start-2]
	ustartm1 = self.u[start-1]
	ustart = self.u[start]
	ustartp1 = self.u[start+1]
	ustartp2 = self.u[start+2]
	ustartp3 = self.u[start+3]

	if wavespeed > 0:
            a = ustartm1 - ustartm2
            b = ustart - ustartm1
            c = ustartp1 - ustart
            d = ustartp2 - ustartp1
            epsilon = 0.000001
            
            IS0 = 13*(a-b)*(a-b) + 3*(a-3*b)*(a-3*b)
            IS1 = 13*(b-c)*(b-c) + 3*(b+c)*(b+c)
            IS2 = 13*(c-d)*(c-d) + 3*(3*c-d)*(3*c-d)
            
            alpha_0 = 1.0 / ((epsilon + IS0) * (epsilon + IS0))
            alpha_1 = 6.0 / ((epsilon + IS1) * (epsilon + IS1))
            alpha_2 = 3.0 / ((epsilon + IS2) * (epsilon + IS2))
            
            
            w0 = alpha_0/(alpha_0 + alpha_1 + alpha_2)
            w2 = alpha_2/(alpha_0 + alpha_1 + alpha_2)
            
            rest = w0*(a - 2*b + c)/3 + (w2 - 0.5)*(b - 2*c + d)/6
        else:
            a = ustartp3 - ustartp2
            b = ustartp2 - ustartp1
            c = ustartp1 - ustart
            d = ustart - ustartm1
            epsilon = 0.000001
            IS0 = 13*(a-b)*(a-b) + 3*(a-3*b)*(a-3*b)
            IS1 = 13*(b-c)*(b-c) + 3*(b+c)*(b+c)
            IS2 = 13*(c-d)*(c-d) + 3*(3*c-d)*(3*c-d)
            
            alpha_0 = 1.0/((epsilon + IS0)*(epsilon + IS0))
            alpha_1 = 6.0/((epsilon + IS1)*(epsilon + IS1))
            alpha_2 = 3.0/((epsilon + IS2)*(epsilon + IS2))
            w0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2)
            w2 = alpha_2 / (alpha_0 + alpha_1 + alpha_2)
            rest = -w0 * (a - 2*b + c) / 3 - (w2 - 0.5) * (b - 2*c + d) / 6
                
	
	u_rec = (-ustartm1 + 7*ustart + 7*ustartp1 - ustartp2) / 12 - rest
	return u_rec

    def WENO7_p(self, i, wavespeed):
        i = i-1
        if wavespeed > 0:
            uim3 = self.u[i-3]
            uim2 = self.u[i-2]
            uim1 = self.u[i-1]
            ui = self.u[i]
            uip1 = self.u[i+1]
            uip2 = self.u[i+2]
            uip3 = self.u[i+3]
        else:
            uim3 = self.u[i+4]
            uim2 = self.u[i+3]
            uim1 = self.u[i+2]
            ui = self.u[i+1]
            uip1 = self.u[i]
            uip2 = self.u[i-1]
            uip3 = self.u[i-2]

        ul = zeros(4)
	ul[0] = - (1.0/4.0)*uim3 + (13.0/12.0)*uim2 - (23.0/12.0)*uim1 + (25.0/12.0)*ui
	ul[1] = (1.0/12.0)*uim2 - (5.0/12.0)*uim1 + (13.0/12.0)*ui + (1.0/4.0)*uip1
	ul[2] =  - (1.0/12.0)*uim1 + (7.0/12.0)*ui + (7.0/12.0)*uip1 - (1.0/12.0)*uip2
	ul[3] = (1.0/4.0)*ui + (13.0/12.0)*uip1 - (5.0/12.0)*uip2 + (1.0/12.0)*uip3

	IS = zeros(4)
	IS[0] = uim3*(547*uim3-3882*uim2+4642*uim1-1854*ui) + uim2*(7043*uim2-17246*uim1+7042*ui) + uim1*(11003*uim1-9402*ui) + 2107*ui*ui
	IS[1] = uim2*(267*uim2-1642*uim1+1602*ui-494*uip1) + uim1*(2843*uim1-5966*ui+1922*uip1) + ui*(3443*ui-2522*uip1) + 547*uip1*uip1
	IS[2] = uim1*(547*uim1-2522*ui+1922*uip1-494*uip2) + ui*(3443*ui-5966*uip1+1602*uip2) + uip1*(2843*uip1-1642*uip2) + 267*uip2*uip2
	IS[3] = ui*(2107*ui-9402*uip1+7042*uip2-1854*uip3) + uip1*(11003*uip1-17246*uip2+4642*uip3) + uip2*(7043*uip2-3882*uip3) + 547*uip3*uip3

	C = zeros(4)
	C[0] = 1.0 / 35.0
	C[1] = 12.0 / 35.0
	C[2] = 18.0 / 35.0
	C[3] = 4.0 / 35.0

	eps = 0.000001	
	alpha = zeros(4)
	alpha_sum = 0.0

	for ci in range(4):
            alpha[ci] = C[ci] / ((eps + IS[ci]) * (eps + IS[ci]))
            alpha_sum += alpha[ci]
            
        w = zeros(4)
        for ci in range(4):
            w[ci] = alpha[ci] / alpha_sum
            

	u_rec = 0.0
	for ci in range(4):
            u_rec += w[ci] * ul[ci]
	return u_rec
    
    def WENO9_p(self, i, wavespeed):
        i = i-1

	if wavespeed > 0:
            uim4 = self.u[i-4]
            uim3 = self.u[i-3]
            uim2 = self.u[i-2]
            uim1 = self.u[i-1]
            ui = self.u[i]
            uip1 = self.u[i+1]
            uip2 = self.u[i+2]
            uip3 = self.u[i+3]
            uip4 = self.u[i+4]
	else:
            uim4 = self.u[i+5]
            uim3 = self.u[i+4]
            uim2 = self.u[i+3]
            uim1 = self.u[i+2]
            ui = self.u[i+1]
            uip1 = self.u[i]
            uip2 = self.u[i-1]
            uip3 = self.u[i-2]
            uip4 = self.u[i-3]

        u_rec = 0

        ul = zeros(5)
        
	ul[0] = (1.0/5.0)*uim4 - (21.0/20.0)*uim3 + (137.0/60.0)*uim2 - (163.0/60.0)*uim1 + (137.0/60.0)*ui
	ul[1] = - (1.0/20.0)*uim3 + (17.0/60.0)*uim2 - (43.0/60.0)*uim1 + (77.0/60.0)*ui + (1.0/5.0)*uip1
	ul[2] = (1.0/30.0)*uim2 - (13.0/60.0)*uim1 + (47.0/60.0)*ui + (9.0/20.0)*uip1 - (1.0/20.0)*uip2
	ul[3] = - (1.0/20.0)*uim1 + (9.0/20.0)*ui + (47.0/60.0)*uip1 - (13.0/60.0)*uip2 + (1.0/30.0)*uip3
	ul[4] = (1.0/5.0)*ui + (77.0/60.0)*uip1 - (43.0/60.0)*uip2 + (17.0/60.0)*uip3 - (1.0/20.0)*uip4

	IS = zeros(5)
	IS[0] = uim4*(22658*uim4-208501*uim3+364863*uim2-288007*uim1+86329*ui) + uim3*(482963*uim3-1704396*uim2+1358458*uim1-411487*ui) + uim2*(1521393*uim2-2462076*uim1+758823*ui) + uim1*(1020563*uim1-649501*ui) + 107918*ui*ui
	IS[1] = uim3*(6908*uim3-60871*uim2+99213*uim1-70237*ui+18079*uip1) + uim2*(138563*uim2-464976*uim1+337018*ui-88297*uip1) + uim1*(406293*uim1-611976*ui+165153*uip1) + ui*(242723*ui-140251*uip1) + 22658*uip1*uip1
	IS[2] = uim2*(6908*uim2-51001*uim1+67923*ui-38947*uip1+8209*uip2) + uim1*(104963*uim1-299076*ui+179098*uip1-38947*uip2) + ui*(231153*ui-299076*uip1+67923*uip2) + uip1*(104963*uip1-51001*uip2) + 6908*uip2*uip2
	IS[3] = uim1*(22658*uim1-140251*ui+165153*uip1-88297*uip2+18079*uip3) + ui*(242723*ui-611976*uip1+337018*uip2-70237*uip3) + uip1*(406293*uip1-464976*uip2+99213*uip3) + uip2*(138563*uip2-60871*uip3) + 6908*uip3*uip3
	IS[4] = ui*(107918*ui-649501*uip1+758823*uip2-411487*uip3+86329*uip4) + uip1*(1020563*uip1-2462076*uip2+1358458*uip3-288007*uip4) + uip2*(1521393*uip2-1704396*uip3+364863*uip4) + uip3*(482963*uip3-208501*uip4) + 22658*uip4*uip4

        C = zeros(5)
	C[0] = 1.0 / 126.0
	C[1] = 10.0 / 63.0
	C[2] = 10.0 / 21.0
	C[3] = 20.0 / 63.0
	C[4] = 5.0 / 126.0

	eps = 0.000001
	alpha = zeros(5)
	alpha_sum = 0.0
	for ci in range(5):
            alpha[ci] = C[ci] / ((eps + IS[ci]) * (eps + IS[ci]))
            alpha_sum += alpha[ci]


        w = zeros(5)
        for ci in range(5):
            w[ci] = alpha[ci] / alpha_sum
            

	u_rec = 0.0
        for ci in range(5):
            u_rec += w[ci] * ul[ci]

	return u_rec



    def roe_scheme(self, i, vm, vp):
        abar = self.f.speed(self.u[i], self.u[i-1])
        if abar >= 0:
            flux = self.f(vp)
        elif abar < 0:
            flux = self.f(vm)
        return flux

    def llf_scheme(self, i, vl, vr):
        alpha = max(abs(self.f.cspeed(vl)),abs(self.f.cspeed(vr)))
        flux = 0.5*((self.f(vl) + self.f(vr)) + alpha*(vl-vr))
        return flux

    def calc_residue(self):
        flux = np.zeros(self.N)
        res = np.zeros(self.N)
        
        #this thing can be parrallelised
        for i in range(self.order, self.N - self.order):
            v = self.calc_val(i)
            vl = v[0]
            vr = v[1]
            if self.scheme == 1:
                flux[i] = self.roe_scheme(i, vl, vr)
            elif self.scheme == 2:
                flux[i] = self.llf_scheme(i, vl, vr)
        
        for i in range(self.order, self.N - self.order-1):
            dx = self.x[i+1]-self.x[i]
            res[i] = -(flux[i+1] - flux[i])/dx
        # boundary periodic
        # for i in range(self.N - self.order-1, self.N):
        #     res[i] = res[self.N - self.order-2]
        # for i in range(self.order):
        #     res[i] = res[self.N-1]
        return res
    

    def calc_dt(self):
        umax = max(abs(self.f.cspeed(self.u)))
        self.dt = self.CFL*(self.x[1]-self.x[0])/umax
        return self.CFL*(self.x[1]-self.x[0])/umax
    

if __name__ == "__main__":
    f = Advection()
    ifunc = Complex()
    N = 1000
    tf = 1.0
    scheme = 2
    xlim = 10
    shift = True
    s = Solver(f = f, xlim=xlim, N=N, CFL=0.8, order=1, ifunc=ifunc, scheme=scheme)
    u_actual = ifunc.init(s.x, tf)
    plot(s.x, u_actual, 'x-')
    I = RK2(solver=s, tf=tf, shift=shift)
    I.run()
    plot(s.xc, s.u, 'x-')
    
    s = Solver(f = f, xlim=xlim, N=N, CFL=0.2, order=2, ifunc=ifunc, scheme=scheme)
    I = RK2(solver=s, tf=tf, shift=shift)
    I.run()
    plot(s.xc, s.u, 'x-')

    s = Solver(f = f, xlim=xlim, N=N, CFL=0.2, order=3, ifunc=ifunc, scheme=scheme)
    I = RK2(solver=s, tf=tf, shift=shift)
    I.run()
    plot(s.xc, s.u, 'x-')

    s = Solver(f = f, xlim=xlim, N=N, CFL=0.1, order=4, ifunc=ifunc, scheme=scheme)
    I = RK2(solver=s, tf=tf, shift=shift)
    I.run()
    plot(s.xc, s.u, 'x-')

    l = ['Actual','Upwind','ENO2','ENO3','ENO4']
    title('Advection equation with RK2 time integration')
    legend(l)

    # s = Solver()
    # s.setup()
    # s.CFL = 0.4
    # I = RK2(solver=s, tf=0tf)
    # I.run()
    # plot(s.xc, s.u, 'x-')
    # # s.setup()
    # legend(['Initial','RK2','RK2'])
    
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
