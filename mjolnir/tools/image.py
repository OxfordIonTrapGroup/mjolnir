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


def check_fit_plausible(y):
    if np.amin(y) * 10 > np.amax(y):
        raise RuntimeError("Insufficient contrast")
    return True


def find_maximum(y):
    """Find maximum pixel location"""
    return {'x0': np.unravel_index(np.argmax(y), y.shape)}


def downsample(img, dwnsmp, pxmap=None):
    """Use averaged binning to downsample an image, returning downsampled
    image and the pixel map giving the coordinates of each binned pixel.

    :param img: image to downsample, as 2D numpy array
    :param dwnsmp: downsampling factor
    :param pxmap: optional pixel coordinate mapping of image, to be binned
        as well. Otherwise assumes integer pixel numbers indexed from 0 for
        the input image.

    :returns: img, pxmap: the binned image and pixel coordinate map of the
        binned pixel centres
    """
    if np.any(np.array(img.shape) % dwnsmp):
        raise ValueError("Downsampling factor must divide image dimensions!")

    sh = (img.shape[0] // dwnsmp, dwnsmp, img.shape[1] // dwnsmp, dwnsmp)
    binned_img = img.reshape(sh).mean(-1).mean(1)

    if pxmap is not None:
        binned_pxmap = pxmap.reshape((2,) + sh).mean(-1).mean(2)
    else:
        binned_pxmap = (np.mgrid[0:img.shape[0]:dwnsmp, 0:img.shape[1]:dwnsmp]
            + (dwnsmp - 1)/2)

    return binned_img, binned_pxmap


def crop(img, centre, region, pxmap=None, dwnsmp=None):
    """Crop an image given centroid and region, returning the cropped image
    and the pixel map giving the coordinates of each pixel.

    All parameters (apart from image values) should be given as integers, and
    pixel map (if given) should be in increasing order.

    :param img: image to crop, as 2D numpy array
    :param centre: coordinates of centre of cropped region. Will be cast to
        integer.
    :param region: size of cropped region. Will be internally truncated to
        nearest even number.
    :param pxmap: optional pixel coordinate mapping of image, to be cropped
        as well. Otherwise assumes integer pixel numbers indexed from 0 for
        the input image.
    :param dwnsmp: optional downsampling factor to consider when cropping.
        Ensure the cropped region is an integer multiple of the
        downsampling factor.

    :returns: img, pxmap: the cropped image, plus its pixel coordinate map
    """
    if pxmap is None:
        pxmap = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    llim = pxmap[:, 0, 0]
    ulim = pxmap[:, -1, -1]

    # Do we actually want this check?
    if np.any(centre < llim) or np.any(centre > ulim):
        raise ValueError("Centre is outside image")

    if dwnsmp is None:
        hr = region // 2
    elif dwnsmp % 2:
        hr = region // (2 * dwnsmp) * dwnsmp
    else:
        hr = region // dwnsmp * dwnsmp / 2

    # Work in pixel coordinates, not using mapping
    pxc = (centre - llim).astype(int)

    lims = np.array([pxc - hr, pxc + hr]).astype(int)
    lims = np.clip(lims, [0, 0], np.array(img.shape) - 1)

    diffs = np.diff(lims, axis=0)[0]
    if not np.all(diffs):
        raise ValueError("Crop parameters caused zero size cropped image")

    if dwnsmp:
        for j, diff in enumerate(diffs):
            if diff % dwnsmp:
                # already forced region to be divisible by downsampling
                # factor, so we must have run into the clip at the edges
                diff = diff // dwnsmp * dwnsmp
                if lims[0,j] == 0:
                    lims[1,j] = diff
                elif lims[1,j] == img.shape[1] - 1:
                    lims[0,j] = lims[1,j] - diff

    xmin, xmax, ymin, ymax = lims.T.flatten()

    cropped_img = img[xmin:xmax, ymin:ymax]
    cropped_pxmap = pxmap[:, xmin:xmax, ymin:ymax]

    return cropped_img, cropped_pxmap


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
    # equivalent to np.moveaxis(x, 0, -1)
    x = np.einsum("i...->...i", x)
    d = x - x0

    # 1.2x speedup possible with ['einsum_path', (0, 1), (0, 1)]
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
    def fit(cls, img, crp=None, dwnsmp=None, pxmap=None):
        """Least squares fit function which can crop and downsample.

        This is the function to use going forwards.
        """
        if crp is not None and dwnsmp is not None:
            assert crp % dwnsmp == 0
        elif dwnsmp is not None:
            assert not np.any(np.array(img.shape) % dwnsmp)

        p0 = find_maximum(img)
        if crp is not None:
            img, pxmap = crop(img, p0['x0'], crp, pxmap=pxmap, dwnsmp=dwnsmp)

        if dwnsmp is not None:
            img, pxmap = downsample(img, dwnsmp, pxmap=pxmap)

        p0 = unpack(parameter_initialiser(pxmap, img))

        check_shape(pxmap, img)
        xdata = pxmap.reshape(2, -1)
        ydata = img.reshape(-1)     #just flatten

        p_fit, p_err = curve_fit(fitting_function, xdata, ydata, p0=p0)
        p = pack(p_fit)
        p.update(cls.compute_derived_properties(p))
        return p

    @classmethod
    def one_step_MLE(cls, xdata, ydata):
        """Calculates MLE of fit parameters on full image.

        Is heavily skewed by background - avoid using on the full image!
        """
        p = parameter_initialiser(xdata, ydata)
        p.update(cls.compute_derived_properties(p))
        return p

    @classmethod
    def two_step_MLE(cls, xdata, ydata, region=50):
        """Calculates MLE on image cropped around maximum pixel value.

        Fast, but more error than the `lsq_cropped` method.
        """
        p = find_maximum(ydata)

        xcrop, ycrop = cls.crop(xdata, ydata, p['x0'], region=region)
        p = parameter_initialiser(xcrop, ycrop)
        p.update(cls.compute_derived_properties(p))
        return p

    @classmethod
    def lsq_fit(cls, xdata, ydata, p0_dict=None, **kwargs):
        """Least squares fit on full image.

        If no initial estimate is provided, will use the one_step_MLE
        to estimate parameters.
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
        """Least squares fit on image cropped around maximum pixel value.

        Fast enough, and accurate.
        """
        p = find_maximum(ydata)

        xcrop, ycrop = cls.crop(xdata, ydata, p['x0'], region=region)
        return cls.lsq_fit(xcrop, ycrop, p0_dict=None)

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
