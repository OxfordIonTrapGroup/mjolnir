# Overall TODO:
#  - rename variables to make it clear what is the image, and what is the
#    pixel mapping
import numpy as np
from scipy.optimize import curve_fit


__all__ = [
    "GaussianBeam",
    "downsample",
    "crop",
    "centred_crop",
    "auto_crop",
]


"""
Note about 2x2 determinant/inverse calculation: obviously there are library
methods to do this, however, with the 2x2 case the algorithm is trivial,
rather than relying on a (complex) generic NxN matrix algorithm.
"""
def det2x2(m):
    """Calculate 2x2 determinant of a matrix

    :param m: 2x2 numpy array to calculate determinant of

    :returns: det, determinant
    """
    assert m.shape == (2,2)
    return m[0,0]*m[1,1] - m[0,1]*m[1,0]


def inv2x2(m):
    """Calculate 2x2 matrix inverse of a matrix

    :param m: 2x2 numpy array to invert

    :returns: inv, the inverse of m sunch that inv@m = eye
    """
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
    p['pxc'] = np.array(args[0:2])
    p['cov'] = np.empty((2,2))
    p['cov'][0,:] = args[2:4]
    p['cov'][1,:] = args[3:5]
    p['scale'] = args[5]
    p['offset'] = args[6]

    return p


def unpack(p):
    """unpack parameter dict into individual parameters"""
    args = np.ones(7)
    args[0:2] = p['pxc']
    args[2:4] = p['cov'][0,:]
    args[4] = p['cov'][1,1]
    args[5] = p['scale']
    args[6] = p['offset']

    return args


def check_shape(pxmap, img):
    _, m, n = pxmap.shape
    assert _ == 2
    assert img.shape == (m, n)


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
        raise ValueError(
            "Downsampling factor {} must divide image dimensions {}"
            .format(dwnsmp, img.shape))

    sh = (img.shape[0] // dwnsmp, dwnsmp, img.shape[1] // dwnsmp, dwnsmp)
    binned_img = img.reshape(sh).mean(-1).mean(1)

    if pxmap is not None:
        binned_pxmap = pxmap.reshape((2,) + sh).mean(-1).mean(2)
    else:
        binned_pxmap = (np.mgrid[0:img.shape[0]:dwnsmp, 0:img.shape[1]:dwnsmp]
            + (dwnsmp - 1)/2)

    return binned_img, binned_pxmap


def crop(img, lims):
    """Crop an image or pixel map acoording to the limits given

    :param img: image or pixel map. Cropping occurs on the last two
        axes, so a pixel map with 3 axes can be supplied
    :param lims: 2x2 array of the limits to crop to, as [[lower], [upper]]

    :returns: cropped array
    """
    x, y = map(slice, *lims)
    return img[..., x, y]


def centred_crop(img, centre, region, pxmap=None):
    """Crop an image given centroid and region, returning the cropped image
    and the pixel map giving the coordinates of each pixel.

    :param img: image to crop, as 2D numpy array
    :param centre: preferred coordinates of centre of cropped region. If this
        would put any of the region over the edges of the image, sets the
        region to the edge.
    :param region: size of cropped region in pixels
    :param pxmap: optional pixel coordinate mapping of image, to be cropped
        as well. Otherwise assumes integer pixel numbers indexed from 0 for
        the input image.

    :returns: img, pxmap: the cropped image, plus its pixel coordinate map

    Note: this is currently messy: centre is given relative to the pixel
    map (if provided), while region is pixel units only. We should either
    make it fully compatible with all dimensions relative to the pixel map,
    or make it pixel units only. The use case of non pixel units only really
    applies if you have already downsampled the image, so it seems like the
    thing to do is support pixel units only.
    """
    region = int(region)
    if np.any(np.array(img.shape) < region):
        raise ValueError(
            "Region {} cannot be larger than image {}"
            .format(region, img.shape))

    if pxmap is None:
        pxmap = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    llim = pxmap[:, 0, 0]
    ulim = llim + img.shape

    # Do we actually want this check?
    if np.any(centre < llim) or np.any(centre > ulim):
        raise ValueError(
            "Centre {} is outside image {}".format(centre, img.shape))

    # Get the first pixel where the pixel map value is equal to or greater
    # than the centre value
    pxc = np.argwhere(
        np.all(np.moveaxis(pxmap, 0, -1) >= centre, axis=-1))[0].astype(int)

    lims = np.array([pxc - (region // 2), pxc + (region // 2) + (region % 2)])
    lims = lims.astype(int)
    lims = np.clip(lims, llim, ulim)

    # I think there may be a better algorithm for doing this; haven't thought
    # of it yet though
    diffs = np.diff(lims, axis=0)[0]
    for j, diff in enumerate(diffs):
        if diff != region:
            if lims[0,j] == llim[j]:
                lims[1,j] = lims[0,j] + region
            elif lims[1,j] == ulim[j]:
                lims[0,j] = lims[1,j] - region

    cropped_img = crop(img, lims)
    cropped_pxmap = crop(pxmap, lims)

    return cropped_img, cropped_pxmap


def auto_crop(img, pxmap=None, dwnsmp_size=None, fill_target=0.3):
    """Auto crop an image for good fitting, with optional downsample for faster
    fitting.

    :param img: image to crop
    :param pxmap: optional pixel map to be cropped. Otherwise assume integer
        numbers indexed from zero.
    :param dwnsmp_size: optional downsampling choice, gives the maximum
        size of the downsampled image. Use None for no downsampling
    :returns: img, pxmap; cropped (and potentially downsampled) image, and
        the corresponding pixel map
    """
    max_ = np.amax(img)
    min_ = np.amin(img)
    contrast = (max_ - min_)
    centre = np.unravel_index(np.argmax(img), img.shape)

    def fill_factor(img):
        """Calculate what fraction of pixels are above dark value"""
        return np.mean(img > min_ + np.exp(-2) * contrast)

    def downsampling_factor(region):
        """Find downsampling factor for approximate region size given"""
        if dwnsmp_size is not None:
            dwnsmp = int(region // dwnsmp_size)
        else:
            dwnsmp = 1
        return dwnsmp if dwnsmp else 1

    def new_region(old, fill):
        """Find next crop region size, and downsampling factor if needed"""
        new = old * np.sqrt(fill / fill_target)
        new = np.clip(new, 0, np.amin(img.shape))

        dwnsmp = downsampling_factor(new)

        if new // dwnsmp == old // dwnsmp:
            new = new // dwnsmp * dwnsmp + np.sign(new - old) * dwnsmp
        else:
            new = new // dwnsmp * dwnsmp
        return new, dwnsmp

    # Hack to get the biggest possible image and downsampling factor
    # (finding the right downsampling for a totally uncropped image
    # is slightly more difficult)
    region, dwnsmp = new_region(np.amin(img.shape), fill_target)
    best = None
    for i in range(5):
        try:
            # This is guaranteed *not* to fail on the first iteration,
            # since our first crop is always the largest feasible crop,
            # so will be neither larger than the image or zero sized
            crp, px = centred_crop(img, centre, region, pxmap=pxmap)
        except ValueError:
            # Either the crop was larger than the image or zero sized,
            # so reset dwnsmp to last successful
            dwnsmp = last
            break
        fill = fill_factor(crp)

        distance = np.abs(fill - fill_target)
        if distance < 0.05:
            break
        elif best is None or distance < best:
            best = distance
            best_effort = (crp, px, dwnsmp)
        last = dwnsmp
        region, dwnsmp = new_region(region, fill)
    else:
        # Didn't break out of the for loop, so use best effort
        crp, px, dwnsmp = best_effort

    if dwnsmp is not None:
        # downsample after choosing size
        crp, px = downsample(crp, dwnsmp, pxmap=px)

    return crp, px


def parameter_initialiser(pxmap, img):
    """naively calculate centroid and covariance of data"""
    # pxmap is like np.mgrid[0:m,0:n]
    check_shape(pxmap, img)

    offset = np.amin(img)
    scale = np.sum(img - offset)
    prob = (img - offset)/scale

    # estimated centre of beam with a weighted mean of each pixel position
    # with its relative intensity
    pxc = np.einsum("ijk,jk->i", pxmap, prob)

    # reshape x to find distances from estimated centre
    # equivalent to np.moveaxis(pxmap, 0, -1)
    pxmap = np.einsum("i...->...i", pxmap)
    d = pxmap - pxc

    # 1.2x speedup possible with ['einsum_path', (0, 1), (0, 1)]
    cov = np.einsum("ijk,ij,ijl->kl", d, prob, d)

    p = {}
    p['pxc'] = pxc
    p['cov'] = cov
    p['scale'] = scale
    p['offset'] = offset

    return p


def _fitting_function(pxmap, p):
    pxmap = np.einsum("i...->...i", pxmap)
    d = pxmap - p['pxc']

    det = det2x2(p['cov'])
    inv = inv2x2(p['cov'])

    pref = (1/(2*np.pi*np.sqrt(np.abs(det))))
    expo = -0.5*np.einsum("...k,kl,...l->...", d, inv, d)

    y = p['scale']*pref*np.exp(expo) + p['offset']

    return y


# annoying function signature for curve_fit
def fitting_function(x, pxc_0=0, pxc_1=0, cov_00=1, cov_01=0, cov_11=1, scale=1, offset=0):
    p = pack([pxc_0, pxc_1, cov_00, cov_01, cov_11, scale, offset])
    return _fitting_function(x, p)


class GaussianBeam:
    @classmethod
    def f(cls, pxmap, p):
        """Return points on a 2D gaussian given by parameters p

        :param pxmap: array of pixel map points to calculate value at
        :param p: parameter dictionary
                  - pxc: centroid
                  - cov: covariance matrix
                  - scale: intensity of 'beam'
                  - offset: image background
        """
        return _fitting_function(pxmap, p)

    @classmethod
    def fit(cls, img, pxmap=None):
        """Fits a gaussian beam with least squares.

        Note: this is extremely slow on large images, so images should be
        cropped and downsampled appropriately before attempting to fit if
        responsiveness is desired. The fitting is susceptible to pixel
        noise if the fitting region is much larger than the beam, so aim
        for the beam to fill approximately 50% of the image.

        :param img: image to fit
        :param pxmap: optional pixel map for the image. otherwise assumes
            integer numbers indexed from 0

        :returns: p, dictionary of fitted and derived parameters

        TODO: Return errors.
        """
        if pxmap is None:
            pxmap = np.mgrid[0:img.shape[0], 0:img.shape[1]]

        p0 = unpack(parameter_initialiser(pxmap, img))

        check_shape(pxmap, img)
        xdata = pxmap.reshape(2, -1)
        ydata = img.reshape(-1)     #just flatten

        p_fit, p_err = curve_fit(fitting_function, xdata, ydata, p0=p0)
        p = pack(p_fit)
        p.update(cls.compute_derived_properties(p))
        return p

    @classmethod
    def compute_derived_properties(cls, p):
        """Calculates useful properties from fitted parameter dict.

        :param p: fitted parameter dictionary
        :returns: params, dictionary of derived parameters
        """
        cov = p['cov']

        params = {}
        params['x_radius'], params['y_radius'] = cls.variance_to_e_squared(
            np.diag(cov))

        # Find eigenvectors/eigenvalues for general ellipse
        w, v = np.linalg.eigh(cov)
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

    @classmethod
    def compute_derived_errors(cls, p, p_err):
        """Not yet implemented: compute the errors of the derived properties.

        Computing the errors on eigenvectors/eigenvalues calculated from an
        uncertain matrix is non-trivial. We could use bootstrapping, i.e.
        generate many matrices according to the distributions of each element,
        calculate the eigenvectors/eigenvalues for each matrix, and then look
        at the distribution of what was returned. Clearly this is
        computationally expensive, so don't do this by default.

        There is also a module on pypi named 'uncertainties' that claims to
        do this for many functions; it's not clear if this is still
        actively developed/maintained.
        """
        raise NotImplementedError
