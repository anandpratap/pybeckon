
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

