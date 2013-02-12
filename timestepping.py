import numpy as np

class Euler:
    def __init__(self, solver,tf):
        self.solver = solver
        self.tf = tf

    def integrate(self, dt, residue):
        pass

    def run(self):
        self.solver.setup()
        t = 0.0
        while t < self.tf:
            dt = self.solver.calc_dt()
            residue = self.solver.calc_residue()
            for i in range(self.solver.N):
                self.solver.u[i] += residue[i]*dt
            t += dt
            print "Time: %s"%(t)

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
        
        self.solver.setup()
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
    
