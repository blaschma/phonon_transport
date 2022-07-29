import numpy as np


class Chain1D:

    def __init__(self, k, N):
        self.k = k
        self.N = N
        self.hessian = self.build_hessian()

    def build_hessian(self):
        hessian = np.zeros((self.N,self.N), dtype=float)

        for i in range(0,self.N):
            #offdiagonals
            if(i<self.N-1):
                print(i)
                hessian[i,i+1] = -self.k
            if(i>0):
                hessian[i, i - 1] = -self.k
            #diagonals
            hessian[i,i] = -np.sum(hessian[i,:])
        return hessian

if __name__ == '__main__':
    model_system = Chain1D(k=0.1, N =3)
    print(model_system.hessian)



