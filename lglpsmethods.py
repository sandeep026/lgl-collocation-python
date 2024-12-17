import numpy as np

''' 
code adapted from supplementary MATLAB code  provided in

Spectral Methods: Algorithms, Analysis and Applications
Jie Shen, Tao Tang and Li-Lian Wang
'''

class LGL:

    def __init__(self, N):
        self.N = N
        self.legslb()
        self.legslbdiff()

    def legslb(self):
        '''
        x=legslb(n) returns n Legendre-Gauss-Lobatto points with x(1)=-1, x(n)=1
        [x,w]= legslb(n) returns n Legendre-Gauss-Lobatto points and weights
        Newton iteration method is used for computing nodes
        Use the function: lepoly() 
        '''
        n = self.N
        # Compute the initial guess of the interior LGL points
        nn = n-1
        thetak = (4*np.linspace(1, nn, nn)-1)*np.pi/(4*nn+2)
        # define both axes for array
        thetak = thetak.reshape(1, thetak.size)
        sigmak = -(1-(nn-1)/(8*nn**3)-(39-28/np.sin(thetak)**2) /
                   (384*nn**4))*np.cos(thetak)
        ze = (sigmak[:, 0:nn-1]+sigmak[:, 1:nn+1])/2
        # error tolerance for stopping iteration
        ep = np.finfo(float).eps*10
        ze1 = ze+ep+1

        # Newton's iteration procedure
        while np.max(np.abs(ze1-ze)) >= ep:
            ze1 = ze
            (dy, y) = self.lepoly(nn, ze)
            # see Page 99 of the book
            ze = ze-(1-ze*ze)*dy/(2*ze*dy-nn*(nn+1)*y)
            # around 6 iterations are required for n=100

        tau = np.concatenate([np.array([[-1]]), ze, np.array([[1]])], 1).T
        # Use the weight expression (3.188) to compute the weights
        quad = np.concatenate(
            [np.array([[2/(nn*(nn+1))]]), 2/(nn*(nn+1)*y**2), np.array([[2/(nn*(nn+1))]])], 1).T

        self.tau = tau
        self.wi = quad

    def lepoly(self, n, x):
        '''
        lepoly  Legendre polynomial of degree n
        y=lepoly(n,x) is the Legendre polynomial
        The degree should be a nonnegative integer 
        The argument x should be on the closed interval [-1,1]; 
        [dy,y]=lepoly(n,x) also returns the values of 1st-order 
        derivative of the Legendre polynomial stored in dy
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
            # kL_k(x)=(2k-1)xL_{k-1}(x)-(k-1)L_{k-2}(x)
            polyn = ((2*k-1)*x*poly-(k-1)*polylst)/k
            # L_k'(x)=L_{k-2}'(x)+(2k-1)L_{k-1}(x)
            pdern = pderlst+(2*k-1)*poly
            polylst = poly
            poly = polyn
            pderlst = pder
            pder = pdern

        return (pdern, polyn)

    def legslbdiff(self):
        '''
        D=legslbdiff(n,x) returns the first-order differentiation matrix of size
        n by n, associated with the Legendre-Gauss-Lobatto points x, which may be computed by 
        x=legslb(n) or x=legslbndm(n). Note: x(1)=-1 and x(n)=1.
        '''
        n = self.N
        x = self.tau

        if n == 0:
            return None

        xx = x
        (dy, y) = self.lepoly(n-1, xx)
        nx = x.shape
        # y is a column vector of L_{n-1}(x_k)
        if nx[1] > nx[0]:
            y = y.T
            xx = x.T

        # compute L_{n-1}(x_j) (x_k-x_j)/L_{n-1}(x_k)
        # 1/d_{kj} for k not= j (see (3.203))
        D = (xx/y)@y.T-(1/y)@(xx*y).T
        # add the identity matrix so that 1./D can be operated
        D = D+np.eye(n)
        D = 1/D
        # update the diagonal entries
        D = D-np.eye(n)
        D[0, 0] = -n*(n-1)/4
        D[n-1, n-1] = -D[0, 0]

        self.D = D


if __name__=="__main__":
    # plotting results
    import matplotlib.pyplot as plt
    # initial and final point
    x0=0
    xf=2*np.pi
    # grid size
    N=50
    # LGL object which has the diff. matrix, quadrature weights and LGL collocation points
    lgl=LGL(N+1)
    # domain transformation [-1,1]->[x0,xf]
    x=lgl.tau*(xf-x0)/2+(xf+x0)/2
    # sample the function 
    y=np.sin(x)
    # compute derivative from differentiation matrix
    y_dot=2/(xf-x0)*lgl.D@y.reshape(N+1,1)
    # plot function and its numerical and analytical derivative
    plt.plot(x,y,label='sine')
    plt.plot(x,y_dot,'-o',label='cosine (lgl)')
    plt.plot(x,np.cos(x),label='cosine (analytical)')
    #plt.plot(x,y_dot-np.cos(x))
    plt.legend()
    plt.grid()
    plt.show()
    print('quadrature:',lgl.wi.T@y)
else:
    pass