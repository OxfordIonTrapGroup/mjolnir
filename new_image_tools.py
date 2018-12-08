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


def parameter_initialiser(x, y, centroid_only=False):
    """naively calculate centroid and covariance of data"""
    # x is like np.mgrid[0:m,0:n]
    check_shape(x, y)

    if centroid_only:
        # estimate centroid with minimal calculation
        x0 = np.einsum("ijk,jk->i", x, y)/np.sum(y)
        return {'x0': x0}

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
    d = x - p.get('x0', np.zeros(2))

    det = det2x2(p.get('cov', np.eye(2)))
    inv = inv2x2(p.get('cov', np.eye(2)))

    pref = (1/(2*np.pi*np.sqrt(np.abs(det))))
    expo = -0.5*np.einsum("...k,kl,...l->...", d, inv, d)

    y = p.get('scale', 1.)*pref*np.exp(expo) + p.get('offset', 0.)

    return y


# annoying function signature for curve_fit
def fitting_function(x, x0_0=0, x0_1=0, c00=1, c01=0, c11=1, a=1, y0=0):
    p = pack([x0_0, x0_1, c00, c01, c11, a, y0])
    return _fitting_function(x, p)


class GaussianBeam:
    @classmethod
    def f(cls, xdata, p):
        """Return points on a 2D gaussian given by parameters p

        :param xdata: array of pixel map points to calculate value at
        :param p: parameter dictionary
                  - x0: centroid, default (0,0)
                  - cov: covariance matrix, default identity
                  - scale: scale factor, default 1
                  - offset: z offset, default 0
        """
        return _fitting_function(xdata, p)

    @classmethod
    def one_step_MLE(cls, xdata, ydata):
        """Performs MLE of parameters on full image

        Can be skewed by noise far from the centroid, and is slow. Avoid.
        """
        p = parameter_initialiser(xdata, ydata)
        p.update(cls.compute_derived_properties(p))
        return p

    @classmethod
    def two_step_MLE(cls, xdata, ydata, region=50):
        """Estimates centroid, then performs MLE on cropped image

        Fast and fairly accurate - good for real time plotting.
        If cropping is too aggressive, will have systematic error visible in
        residuals.
        """
        p = parameter_initialiser(xdata, ydata, centroid_only=True)

        xcrop, ycrop = cls.crop(xdata, ydata, p['x0'], region=region)
        p = parameter_initialiser(xcrop, ycrop)
        p.update(cls.compute_derived_properties(p))
        return p


    @classmethod
    def lsq_fit(cls, xdata, ydata, p0_dict=None, **kwargs):
        """Least squares fit on all data points

        If no initial estimate is provided, will use the one_step_MLE
        to start with. This is inefficient and should be revised.
        """
        if p0_dict is None:
            p0 = unpack(parameter_initialiser(xdata, ydata))
        else:
            p0 = unpack(p0_dict)

        check_shape(xdata, ydata)
        xdata = xdata.reshape(2, -1)
        ydata = ydata.reshape(-1)     #just flatten

        p_fit, p_err = curve_fit(fitting_function, xdata, ydata, p0=p0)
        p = pack(p_fit)
        p.update(cls.compute_derived_properties(p))
        return p

    @classmethod
    def lsq_cropped(cls, xdata, ydata, region=50):
        """Estimates centroid, then uses least squares on cropped data"""
        p = parameter_initialiser(xdata, ydata, centroid_only=True)

        xcrop, ycrop = cls.crop(xdata, ydata, p['x0'], region=region)
        return self.lsq_fit(xcrop, ycrop, p0_dict=p)

    @classmethod
    def _get_limits(cls, x0, shape, region=50):
        """Returns cropped coordinates (imin, imax, jmin, jmax)

        Will ensure that all points are within bounds.
        """
        hr = int(region/2)
        x0 = np.around(x0).astype(int)
        lims = np.array([x0-hr, x0+hr])
        return np.clip(lims, [0,0], np.array(shape)-1).T.flatten()

    @classmethod
    def crop(cls, xdata, ydata, x0, region=50):
        """Crops both pixel map and image given centroid and region"""
        imin, imax, jmin, jmax = cls._get_limits(x0, ydata.shape, region)
        xcrop = xdata[:, imin:imax, jmin:jmax]
        ycrop = ydata[imin:imax, jmin:jmax]
        return xcrop, ycrop

    @classmethod
    def compute_derived_properties(cls, p):
        """Calculates useful properties from fitted parameter dict"""

        cov = p['cov']

        params = {}
        params['x_radius'], params["y_radius"] = cls.variance_to_e_squared(
            np.diag(cov))

        # Find eigenvectors/eigenvalues for general ellipse
        w, v = np.linalg.eig(cov)
        maj = np.argmax(w)
        min_ = maj - 1     # trick: we can always index with -1 or 0
        params['semimaj_angle'] = np.rad2deg(np.arctan2(v[1, maj], v[0, maj]))
        params['semimin_angle'] = np.rad2deg(np.arctan2(v[1, min_], v[0, min_]))
        params['semimaj'] = cls.variance_to_e_squared(w[maj])
        params['semimin'] = cls.variance_to_e_squared(w[min_])

        # Ellipticity
        params['e'] = 1 - (params['semimin'] / params['semimaj'])

        # Average radius
        # Calculate gemometric mean so that area is the same
        params['avg_radius'] = np.sqrt(params['semimin'] * params['semimaj'])

        return params

    @classmethod
    def variance_to_e_squared(cls, var):
        """Convert variance to 1/e^2 radius"""
        return 2*np.sqrt(var)

gaussian_beam = GaussianBeam()
