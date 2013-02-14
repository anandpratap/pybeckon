import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from progressbar import ProgressBar
import time
from initial import Complex

class Euler:
    def __init__(self, solver, tf, shift=False):
        self.solver = solver
        self.tf = tf
        self.shift = shift
        self.pbar = ProgressBar(maxval = self.tf)
    def integrate(self):
        pass

    def savedata(self):
        # filename format EQUATION_ORDER_SCHEME_N_TIME.npz
        filename = "./data/%s_%s_%s_%s_%s.npz"%(self.solver.f.__name__, self.solver.order, self.solver.scheme, self.solver.N, self.tf)
        np.savez(filename, x = self.solver.xc, u = self.solver.u, uactual = Complex().init(self.solver.xc, self.tf))
        print filename

    def run(self):
        t = 0.0
        while t < self.tf:
            dt = self.solver.calc_dt()
            residue = self.solver.calc_residue()
            for i in range(self.solver.N):
                self.solver.u[i] += residue[i]*dt
            self.pbar.update(t)
            t += dt
            print "Time: %s"%(t)
        self.savedata()

class CudaEuler:
    def __init__(self, solver,tf, shift=False):
        self.solver = solver
        self.tf = tf
        mod = SourceModule("""
           __global__ void integrate(double *dest, double *u, double *residue, double dt)
           {
            const int i = threadIdx.x;
           dest[i] = u[i]  +  residue[i]*dt;
           }
         """)
    
        self.integrate = mod.get_function("integrate")

    def integrate(self, dt, residue):
        pass

    def run(self):
        t = 0.0
        while t < self.tf:
            dt = self.solver.calc_dt().astype(np.float64)
            residue = self.solver.calc_residue().astype(np.float64)
            u = self.solver.u.astype(np.float64)
            un = np.zeros_like(u)
            size = len(u)
            self.integrate(drv.Out(un), drv.In(u), drv.In(residue), dt, block=(size,1,1), grid=(1,1))
            self.solver.u = un
            t += dt
        self.savedata()            
            

class RK2(Euler):
    def run(self):
        alpha = np.zeros([3,3])
        beta = np.zeros([3,3])
        alpha[1,0] = 1.0
        alpha[2,0] = 0.5
        alpha[2,1] = 0.5
        beta[1,0] = 1.0
        beta[2,0] = 0.0
        beta[2,1] = 0.5
        
        t = 0.0
        while t < self.tf:
            dt = self.solver.calc_dt()
            u0 = np.copy(self.solver.u)
            residue0 = self.solver.calc_residue()
            u1 = alpha[1,0]*u0 + beta[1,0]*dt*residue0
            self.solver.u = u1
            residue1 = self.solver.calc_residue()
            self.solver.u = alpha[2,0]*u0 + beta[2,0]*dt*residue0 + alpha[2,1]*u1 + beta[2,1]*dt*residue1
            distance = self.solver.f.a*dt
            N = distance/self.solver.dx
            self.pbar.update(t)
            t += dt
        self.savedata()
            
class RK4(Euler):
    def run(self):
        alpha = np.zeros([3,3])
        beta = np.zeros([3,3])
        alpha[1,0] = 1.0
        alpha[2,0] = 0.5
        alpha[2,1] = 0.5
        beta[1,0] = 1.0
        beta[2,0] = 0.0
        beta[2,1] = 0.5
        
        t = 0.0
        while t < self.tf:
            dt = self.solver.calc_dt()
            u0 = np.copy(self.solver.u)
            residue0 = self.solver.calc_residue()
            u1 = alpha[1,0]*u0 + beta[1,0]*dt*residue0
            self.solver.u = u1
            residue1 = self.solver.calc_residue()
            self.solver.u = alpha[2,0]*u0 + beta[2,0]*dt*residue0 + alpha[2,1]*u1 + beta[2,1]*dt*residue1
            t += dt
            print "Time: %s"%(t)
        self.savedata()
