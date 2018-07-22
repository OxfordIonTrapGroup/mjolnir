import numpy as np
from scipy.optimize import curve_fit


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


def pack(args):
    """packs parameters given as scalars into parameter dict"""
    p = {}
    p['x0'] = np.array(args[0:2])
    p['cov'] = np.empty((2,2))
    p['cov'][0,:] = args[2:4]
    p['cov'][1,:] = args[3:5]
    p['scale'] = args[5]
    p['offset'] = args[6]

    return p


def unpack(p):
    """unpack parameter dict into individual parameters"""
    args = np.ones(7)
    args[0:2] = p['x0']
    args[2:4] = p['cov'][0,:]
    args[4] = p['cov'][1,1]
    args[5] = p['scale']
    args[6] = p['offset']

    return args


def check_shape(x, y):
    # print(x.shape)
    # print(y.shape)
    _, m, n = x.shape
    assert _ == 2
    assert y.shape == (m, n)


def parameter_initialiser(x, y):
    """naively calculate centroid and covariance of data"""
    # x is like np.mgrid[0:m.0:n]
    check_shape(x, y)

    y0 = np.amin(y)
    a = np.sum(y - y0)
    prob = (y - y0)/a

    x0 = np.einsum("ijk,jk->i", x, prob)

    # reshape x to find distances from mean position
    x = np.einsum("i...->...i", x)
    d = x - x0

    cov = np.einsum("ijk,ij,ijl->kl", d, prob, d)

    p = {}
    p['x0'] = x0
    p['cov'] = cov
    p['scale'] = a
    p['offset'] = y0

    return p


def _fitting_function(x, p):
    x = np.einsum("i...->...i", x)
    d = x - p['x0']

    det = det2x2(p['cov'])
    inv = inv2x2(p['cov'])

    pref = (1/(2*np.pi*np.sqrt(np.abs(det))))
    expo = -0.5*np.einsum("...k,kl,...l->...", d, inv, d)

    y = p['scale']*pref*np.exp(expo) + p['offset']

    return y


# annoying function signature for curve_fit
def fitting_function(x, x0_0, x0_1, c00, c01, c11, a, y0):
    p = pack([x0_0, x0_1, c00, c01, c11, a, y0])
    return _fitting_function(x, p)


class GaussianBeam:
    def naive_fit(self, xdata, ydata):
        return parameter_initialiser(xdata, ydata)

    def two_step_fit(self, xdata, ydata, region=50):
        p = parameter_initialiser(xdata, ydata)
        hr = int(region/2)

        ii,jj = np.around(p['x0']).astype(int)
        mask = np.full_like(ydata, False, dtype=bool)
        mask[ii-hr:ii+hr, jj-hr:jj+hr] = True

        masked = np.copy(ydata)
        masked[~mask] = 0.
        return parameter_initialiser(xdata, masked)

    def two_step_fit_mk2(self, xdata, ydata, region=50):
        p = parameter_initialiser(xdata, ydata)
        hr = int(region/2)

        ii,jj = np.around(p['x0']).astype(int)
        xcrop = xdata[:, ii-hr:ii+hr, jj-hr:jj+hr]
        ycrop = ydata[ii-hr:ii+hr, jj-hr:jj+hr]

        return parameter_initialiser(xcrop, ycrop)

    def lsq_fit(self, xdata, ydata, p0_dict=None, **kwargs):
        if p0_dict is None:
            p0 = unpack(parameter_initialiser(xdata, ydata))
        else:
            p0 = unpack(p0_dict)

        check_shape(xdata, ydata)
        xdata = xdata.reshape(2, -1)
        ydata = ydata.reshape(-1)     #just flatten

        p_fit, p_err = curve_fit(fitting_function, xdata, ydata, p0=p0)
        return pack(p_fit)

    def lsq_cropped(self, xdata, ydata, region=50):
        p = parameter_initialiser(xdata, ydata)
        hr = int(region/2)

        ii,jj = np.around(p['x0']).astype(int)
        xcrop = xdata[:, ii-hr:ii+hr, jj-hr:jj+hr]
        ycrop = ydata[ii-hr:ii+hr, jj-hr:jj+hr]

        return self.lsq_fit(xcrop, ycrop, p0_dict=p)


gaussian_beam = GaussianBeam()
