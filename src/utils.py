class NewtonDD:
    """
    Class to be used to calculate Newton's Divided Difference for ENO
    ==================================================================

    Initialize with an array of coordiates, values at those points,
    and flux object.
    
    Then any divided difference can be calculated by calling the calc function.
    
    Example:
    =======
    ---------------------------------------
    import numpy as np
    
    x = np.linspace(0.0,1.0,10)
    y = np.sin(x)
    f = flux.Burgers()
    ndd = NewtonDD(x=x, u=y, f=f)
    
    ddiff = ndd.calc(i=1, j=5)

    --------------------------------------
    Note: i should always be less then j
    --------------------------------------
    """
    
    def __init__(self, x, u, f):
        self.x = x
        self.u = u
        self.f = f

        
    def calc(self, i, j):
        if j - i == 1:
            return self.f(self.u[i])
        else:
            return (self.calc(i, j-1) - self.calc(i+1, j))/(self.x[i] - self.x[j])

    def calcu(self,  i, j):
        if j == i:
            return self.u[i]
        else:
            return self.calcu(i+1, j) - self.calcu(i, j-1)
