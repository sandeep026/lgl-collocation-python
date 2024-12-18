import numpy as np

''' 
code adapted from supplementary MATLAB code provided in
Spectral Methods: Algorithms, Analysis and Applications
Jie Shen, Tao Tang and Li-Lian Wang
'''


class LGL:
    '''
    compute LGL weights, nodes and differentiation matrix
    '''

    def __init__(self, N):
        self.N = N
        self.legslb()
        self.legslbdiff()

    def legslb(self):
        '''
        x Legendre-Gauss-Lobatto points
        w Legendre-Gauss-Lobatto weights/ quadrature
        Newton method to compute nodes 
        '''
        n = self.N
        # initial guess for x
        nn = n-1
        thetak = (4*np.linspace(1, nn, nn)-1)*np.pi/(4*nn+2)
        # define both axes for array
        thetak = thetak.reshape(1, thetak.size)
        sigmak = -(1-(nn-1)/(8*nn**3)-(39-28/np.sin(thetak)**2) /
                   (384*nn**4))*np.cos(thetak)
        ze = (sigmak[:, 0:nn-1]+sigmak[:, 1:nn+1])/2
        # error tolerance
        ep = np.finfo(float).eps*10
        ze1 = ze+ep+1

        # Newton method
        while np.max(np.abs(ze1-ze)) >= ep:
            ze1 = ze
            (dy, y) = self.lepoly(nn, ze)
            # see Page 99 of the book
            ze = ze-(1-ze*ze)*dy/(2*ze*dy-nn*(nn+1)*y)
            # 6 iterations for n=100

        tau = np.concatenate([np.array([[-1]]), ze, np.array([[1]])], 1).T
        # 3.188 from book to compute the quadrature weights
        quad = np.concatenate(
            [np.array([[2/(nn*(nn+1))]]), 2/(nn*(nn+1)*y**2), np.array([[2/(nn*(nn+1))]])], 1).T

        self.tau = tau
        self.wi = quad

    def lepoly(self, n, x):
        '''
        n  polynomial degree 
        x  vector of n elements in [-1,1] 
        y  Legendre polynomial of degree n
        dy 1st derivative of the legendre polynomial
        '''

        if n == 0:
            return (np.zeros(x.shape), np.ones(x.shape))

        if n == 1:
            return (np.ones(x.shape), x)

        polylst = np.ones(x.shape)
        pderlst = np.zeros(x.shape)
        poly = x
        pder = np.ones(x.shape)
        # L_0=1 L_0'=0 L_1=x  L_1'=1
        # Three-term recurrence relation:

        temp = np.linspace(2, n, (n-1))
        for k in temp:
            polyn = ((2*k-1)*x*poly-(k-1)*polylst)/k
            pdern = pderlst+(2*k-1)*poly
            polylst = poly
            poly = polyn
            pderlst = pder
            pder = pdern

        return (pdern, polyn)

    def legslbdiff(self):
        '''
        D Differentiation matrix of size n by n 
        '''
        n = self.N
        x = self.tau

        if n == 0:
            return None

        xx = x
        (dy, y) = self.lepoly(n-1, xx)
        nx = x.shape
        if nx[1] > nx[0]:
            y = y.T
            xx = x.T

        D = (xx/y)@y.T-(1/y)@(xx*y).T
        D = D+np.eye(n)
        D = 1/D
        D = D-np.eye(n)
        D[0, 0] = -n*(n-1)/4
        D[n-1, n-1] = -D[0, 0]
        self.D = D


if __name__ == "__main__":
    # plotting results
    import matplotlib.pyplot as plt
    # initial and final point
    x0 = 0
    xf = 2*np.pi
    # grid size
    N = 50
    # LGL object which has the diff. matrix, quadrature weights and LGL collocation points
    lgl = LGL(N+1)
    # domain transformation [-1,1]->[x0,xf]
    x = lgl.tau*(xf-x0)/2+(xf+x0)/2
    # sample the function
    y = np.sin(x)
    # compute derivative from differentiation matrix
    y_dot = 2/(xf-x0)*lgl.D@y.reshape(N+1, 1)
    # plot function and its numerical and analytical derivative
    plt.plot(x, y, label='sine')
    plt.plot(x, y_dot, '-o', label='cosine (lgl)')
    plt.plot(x, np.cos(x), label='cosine (analytical)')
    # plt.plot(x,y_dot-np.cos(x))
    plt.legend()
    plt.grid()
    plt.show()
    print('quadrature:', lgl.wi.T@y)
else:
    pass
