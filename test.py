import numpy as np
import numpy.testing as npt
import pytest


from lgltools import LGL

# LGL weights and nodes
# dymos unit tests for LGL nodes and weights
# generation of nodes and weighta from Wang
x_i = {2: [-1.0, 1.0],
       3: [-1.0, 0.0, 1.0],
       4: [-1.0, -np.sqrt(5)/5, np.sqrt(5)/5, 1.0],
       5: [-1.0, -np.sqrt(21)/7, 0.0, np.sqrt(21)/7, 1.0],
       6: [-1.0, -np.sqrt((7 + 2 * np.sqrt(7)) / 21), -np.sqrt((7 - 2 * np.sqrt(7)) / 21),
           np.sqrt((7 - 2 * np.sqrt(7)) / 21), np.sqrt((7 + 2 * np.sqrt(7)) / 21), 1.0]}

w_i = {2: [1.0, 1.0],
       3: [1/3, 4/3, 1/3],
       4: [1/6, 5/6, 5/6, 1/6],
       5: [1/10, 49/90, 32/45, 49/90, 1/10],
       6: [1/15, (14 - np.sqrt(7))/30, (14 + np.sqrt(7))/30,
           (14 + np.sqrt(7))/30, (14 - np.sqrt(7))/30, 1/15]}

for i in range(2,7):
    lgl=LGL(i)
    npt.assert_array_almost_equal(lgl.wi, np.array(w_i[i]).reshape(-1,1))
    npt.assert_array_almost_equal(lgl.tau,np.array(x_i[i]).reshape(-1,1))


# data for testing differentiation and integration matrices
x0 = 0
xf = 2 * np.pi
# grid size
N = 50
# LGL object which has the diff. matrix, quadrature weights and LGL collocation points
lgl = LGL(N + 1)
# domain transformation [-1,1]->[x0,xf]
x = lgl.tau * (xf - x0) / 2 + (xf + x0) / 2
# sample the function
y = np.sin(x)

# integral sinx 0-2pi -> zero. quadrature must capture the behaviour
npt.assert_array_almost_equal(
    (x[-1] - x[0]) / 2 * np.dot(lgl.wi.ravel(), y), 0, decimal=4
)

# analytical check on derivative vs Diff. matrix
# compute derivative from differentiation matrix
y_dot = 2 / (xf - x0) * lgl.D @ y.reshape(N + 1, 1)
y_dot_an = np.cos(x)
npt.assert_array_almost_equal(y_dot, y_dot_an, decimal=4)
# Derivative of constant = 0
ones = np.ones(N + 1)
zeros = np.zeros(N + 1)
npt.assert_array_almost_equal(
    lgl.D @ ones.reshape(N + 1, 1), zeros.reshape(N + 1, 1), decimal=4
)

# last row of integration matrix must match lgl quadrature
npt.assert_array_almost_equal(
    (x[-1] - x[0]) / 2 * np.dot(lgl.A[-1, :].ravel(), y), 0, decimal=4
)
npt.assert_array_almost_equal(
    (x[-1] - x[0]) / 2 * np.dot(lgl.Birk[-1, :].ravel(), y), 0, decimal=4
)
