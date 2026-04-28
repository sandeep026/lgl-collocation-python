import numpy.testing as npt
import numpy as np
import pytest

from lglpsmethods import LGL

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
# compute derivative from differentiation matrix
y_dot = 2 / (xf - x0) * lgl.D @ y.reshape(N + 1, 1)
y_dot_an=np.cos(x)
ones=np.ones(N+1)
zeros=np.zeros(N+1)


# integral sinx 0-2pi -> zero
def integral_sin() -> np.ndarray:
    return (x[-1] - x[0]) / 2 * np.dot(lgl.wi.ravel(), y)

npt.assert_array_almost_equal(integral_sin(),0,decimal=4)

npt.assert_array_almost_equal(y_dot,y_dot_an,decimal=4)

npt.assert_array_almost_equal(lgl.D@ones.reshape(N+1,1),zeros.reshape(N+1,1),decimal=4)

