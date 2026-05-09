"""
code adapted from supplementary MATLAB code provided in
Spectral Methods: Algorithms, Analysis and Applications
Jie Shen, Tao Tang and Li-Lian Wang
"""

import numpy as np
from numpy.polynomial.legendre import legvander

class LGL:
    """
    Compute LGL weights, nodes and differentiation matrix
    """

    def __init__(self, N):
        self.N = N
        self.legslb()
        self.legslbdiff()
        self.legslbint()
        self.birkhoff_matrix()

    def legslb(self):
        """
        x Legendre-Gauss-Lobatto points
        w Legendre-Gauss-Lobatto weights/ quadrature
        Newton method to compute nodes
        """
        n = self.N
        # initial guess for x
        nn = n - 1
        thetak = (4 * np.linspace(1, nn, nn) - 1) * np.pi / (4 * nn + 2)
        # define both axes for array
        thetak = thetak.reshape(1, thetak.size)
        sigmak = -(
            1 - (nn - 1) / (8 * nn**3) - (39 - 28 / np.sin(thetak) ** 2) / (384 * nn**4)
        ) * np.cos(thetak)
        ze = (sigmak[:, 0 : nn - 1] + sigmak[:, 1 : nn + 1]) / 2
        # error tolerance
        ep = np.finfo(float).eps * 10
        ze1 = ze + ep + 1

        # Newton method
        if self.N==2:
            tau=np.array([[-1],[1]])
            quad=np.array([[1],[1]])
        else:
            while np.max(np.abs(ze1 - ze)) >= ep:
                ze1 = ze
                (dy, y) = self.lepoly(nn, ze)
                # see Page 99 of the book
                ze = ze - (1 - ze * ze) * dy / (2 * ze * dy - nn * (nn + 1) * y)
                # 6 iterations for n=100
                tau = np.concatenate([np.array([[-1]]), ze, np.array([[1]])], 1).T
                # 3.188 from book to compute the quadrature weights
                quad = np.concatenate(
                    [
                        np.array([[2 / (nn * (nn + 1))]]),
                        2 / (nn * (nn + 1) * y**2),
                        np.array([[2 / (nn * (nn + 1))]]),
                    ],
                    1,
                ).T

        self.tau = tau
        self.wi = quad

    def lepoly(self, n, x):
        """
        n  polynomial degree
        x  vector of n elements in [-1,1]
        y  Legendre polynomial of degree n
        dy 1st derivative of the legendre polynomial
        """
        x = np.atleast_1d(x)
        if n == 0:
            return (np.zeros_like(x), np.ones_like(x))
        if n == 1:
            return (np.ones_like(x), x)
        polylst = np.ones_like(x)
        pderlst = np.zeros_like(x)
        poly = x
        pder = np.ones_like(x)
        # L_0=1 L_0'=0 L_1=x  L_1'=1
        # Three-term recurrence relation:
        for k in range(2, n + 1):
            polyn = ((2 * k - 1) * x * poly - (k - 1) * polylst) / k
            pdern = pderlst + (2 * k - 1) * poly
            polylst = poly
            poly = polyn
            pderlst = pder
            pder = pdern
        return (pdern, polyn)

    def legslbdiff(self):
        """
        D Differentiation matrix of size n by n
        """
        n = self.N
        x = self.tau
        if n == 0:
            return None
        xx = x
        (dy, y) = self.lepoly(n - 1, xx)
        nx = x.shape
        if nx[1] > nx[0]:
            y = y.T
            xx = x.T

        D = (xx / y) @ y.T - (1 / y) @ (xx * y).T
        D = D + np.eye(n)
        D = 1 / D
        D = D - np.eye(n)
        D[0, 0] = -n * (n - 1) / 4
        D[n - 1, n - 1] = -D[0, 0]
        self.D = D

    def legslbint(self):
        """
        Vectorized integration matrix computation of size (N-1) x N
        """

        nodes = self.N
        self.A = np.zeros((nodes - 1, nodes))
        W = self.wi.T.flatten()
        TAU = self.tau.T.flatten()

        # Precompute lepoly values for all j and tau values needed
        J = np.arange(1, nodes + 1)
        L_matrix = np.zeros((len(J), nodes))  # lepoly(j, tau_k)
        for idx, j in enumerate(J):
            L_matrix[idx, :] = self.lepoly(j, TAU[0:nodes])[1]

        # Precompute L_{j+1}(tau_i) and L_{j-1}(tau_i) for all i
        L_plus = np.zeros((len(J), nodes - 1))
        L_minus = np.zeros((len(J), nodes - 1))
        for idx, j in enumerate(J):
            L_plus[idx, :] = self.lepoly(j + 1, TAU[1:nodes])[1]
            L_minus[idx, :] = self.lepoly(j - 1, TAU[1:nodes])[1]

        # Compute the sigma matrix (nodes-1 x nodes)
        SIGMA = np.dot((L_plus - L_minus).T, L_matrix)

        # Broadcast W and TAU to compute A matrix
        for i in range(1, nodes):
            self.A[i - 1, :] = (W / 2.0) * (TAU[i] + 1.0 + SIGMA[i - 1, :])

    def birkhoff_matrix(self) -> np.ndarray:
        tau = self.tau
        w = self.wi
        tau = np.asarray(tau).ravel()
        w = np.asarray(w).ravel()
        N = tau.size - 1

        V = legvander(tau, N + 1)

        alpha = V[:, : N + 1].T * w
        alpha[N, :] *= N / (2 * N + 1)

        S = np.zeros((N + 1, N + 1))
        S[:, 0] = (tau - tau[0]) / 2.0
        S[:, 1:] = (V[:, 2:] - V[:, :N]) / 2.0

        self.Birk = S @ alpha
