
import numpy as np
from scipy.optimize import curve_fit
from image_tools import derived_results


def pack(args):
    """packs parameters given as scalars into parameter dict"""
    p = {}
    p['mean'] = np.array(args[0:2])
    p['cov'] = np.empty((2,2))
    p['cov'][0,:] = args[2:4]
    p['cov'][1,:] = args[3:5]
    p['scale'] = args[5]
    p['offset'] = args[6]

    return p


def unpack(p):
    """unpack parameter dict into individual parameters"""
    args = np.ones(7)
    args[0:2] = p['mean']
    args[2:4] = p['cov'][0,:]
    args[4] = p['cov'][1,1]
    args[5] = p['scale']
    args[6] = p['offset']

    return args


def det2x2(m):
    """2x2 determinant"""
    assert m.shape == (2,2)
    return m[0,0]*m[1,1] - m[0,1]*m[1,0]


def inv2x2(m):
    """2x2 matrix inverse"""
    assert m.shape == (2,2)
    det = det2x2(m)
    assert det != 0

    inv = np.empty((2,2))
    inv[0,0] = m[1,1]/det
    inv[0,1] = -m[0,1]/det
    inv[1,0] = -m[1,0]/det
    inv[1,1] = m[0,0]/det

    assert np.allclose(inv@m, np.eye(2))

    return inv


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

    p = {}
    p['mean'] = x0
    p['cov'] = cov
    p['scale'] = a
    p['offset'] = y0

    return p


def _fitting_function(x, p):
    x0 = p['mean']
    cov = p['cov']
    scale = p['scale']
    y0 = p['offset']

    x = np.einsum("i...->...i", x)
    d = x - x0

    det = det2x2(cov)
    inv = inv2x2(cov)

    pref = (1/(2*np.pi*np.sqrt(np.abs(det))))
    expo = -0.5*np.einsum("...k,kl,...l->...", d, inv, d)

    y = scale*pref*np.exp(expo) + y0

    return y


# annoying function signature for curve_fit
def fitting_function(x, x0_0, x0_1, c00, c01, c11, a, y0):
    p = pack([x0_0, x0_1, c00, c01, c11, a, y0])
    return _fitting_function(x, p)


class gaussian_beam:
    def fit(self, xdata, ydata, p0_dict=None, **kwargs):
        if p0_dict is None:
            p0 = unpack(parameter_initialiser(xdata, ydata))
            print(pack(p0))
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

        p_fit, p_err = curve_fit(fitting_function, xdata, ydata, p0=p0)
        return pack(p_fit)
