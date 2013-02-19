pybeckon
========

Numerical Schemes for 1D wave equations

* Make sure that the src directory is in your python path

```python

	import numpy as np
	from flux import Advection, Burgers
	from timestepping import RK2, Euler
	from initial import Step, InvertedStep, Complex, Square, Sin
	from solver import Solver
	from pylab import *

        # define flux function
	# custom flux function can be defined, see the template in flux.py
	f = Advection(a=1.0)
    	
	# define initial condition
	# custom condition can be defined, seee the template in initial.py 
	ifunc = Complex()

	# number of points
	N = 1000
    
	# final time
	tf = 1.0

	# flux scheme 1 -> Roe 2 -> Local Lax Friedrichs
	scheme = 2
    	
	# domain is from -xlim to xlim
	xlim = 10
	
	# following option is redundant as of now
	shift = True
   
	# define order
	# 1 -> Godunov
	# 2 -> ENO2
	# 3 -> ENO3
	# 4 -> ENO4
	# 5 -> WENO5
	# 7 -> WENO7
	# 9 -> WENO9

	# init solver
	s = Solver(f = f, xlim=xlim, N=N, CFL=0.8, order=order, ifunc=ifunc, scheme=scheme)
    
	# u_actual will only work for advection equation
	u_actual = ifunc.init(s.x, tf)
	
	# plot analytical solution
	plot(s.x, u_actual, 'x-')
    
	# initialize integrator, other integrator available is Euler
	I = RK2(solver=s, tf=tf, shift=shift)

	# run the simulation
	I.run()

	# plot the final results
	plot(s.xc, I.solver.u, 'x-')
```