
import numpy as np
from scipy.optimize import curve_fit


def parameter_initialiser(x, y):
    # x is like np.mgrid[0:m.0:n]
    y0 = np.amin(y)
    a = np.sum(y - y0)
    prob = (y - y0)/a

    x0 = np.einsum("ijk,jk->i", x, prob)

    # reshape x to find distances from mean position
    x = np.einsum("i...->...i", x)
    d = x - x0

    cov = np.einsum("ijk,ij,ijl->kl", d, prob, d)

    return x0[0], x0[1], cov[0,0], cov[0,1], cov[1,1], a, y0


# annoying function signature
def fitting_function(x, x0_0, x0_1, c00, c01, c11, a, y0):
    x0 = np.array([x0_0, x0_1])
    x = np.einsum("i...->...i", x)
    d = x - x0

    # these operations can be done with numpy.linalg, but that's probably very
    # slow. our covariance matrix is symmetric by definition and can easily be
    # inverted manually
    det = c00*c00 - c01*c01
    inv = np.zeros((2,2))
    inv[0,0] = c11/det
    inv[0,1] = -c01/det
    inv[1,0] = inv[0,1]
    inv[1,1] = c00/det

    pref = (1/(2*np.pi*np.sqrt(np.abs(det))))
    expo = -0.5*np.einsum("...k,kl,...l->...", d, inv, d)

    y = a*pref*np.exp(expo) + y0

    return y


class gaussian_beam:
    def fit(self, xdata, ydata, p0_dict=None, **kwargs):
        if p0_dict is None:
            p0 = parameter_initialiser(xdata, ydata)
        else:
            p0 = None
            # p0 = np.zeros(4)
            # p0[0] = p0_dict.get('x0', np.ones(2))
            # p0[1] = p0_dict.get('cov', np.eye(2))
            # p0[2] = p0_dict.get('y0', 0.)
            # p0[3] = p0_dict.get('a', 1.)

        _, m, n = xdata.shape
        assert _ == 2
        xdata = xdata.reshape(2, m*n)
        ydata = ydata.reshape(m*n)     #just flatten

        return curve_fit(fitting_function, xdata, ydata, p0=p0)
