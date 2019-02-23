import numpy as np
from scipy import misc
from scipy.stats import multivariate_normal

from mjolnir.tools.image import *


def generate_image(
        centroid=[640, 512],
        cov=100*np.eye(2),
        intensity=220,
        noise="poisson",
        background=0,
        m=1280,
        n=1024):

    pxmap = np.mgrid[0:m,0:n]
    pxmap = np.einsum("i...->...i", pxmap)
    img = multivariate_normal(centroid, cov).pdf(pxmap)

    # scale it
    img *= intensity/np.amax(img)

    # add background
    img += background

    if noise == "poisson":
        img = np.random.poisson(lam=img)

    return np.clip(img.astype(int), 0, 255)


def load_image(fname):
    img = misc.imread(fname)
    img = img.astype(np.uint8)
    return img.T


def main():
    filenames = [f+".bmp" for f in ["collimated", "collimated1", "focus", "focus1"]]

    for f in filenames:
        img = load_image(f)

        # Fitting whole image
        p = GaussianBeam.fit(img)

        # Croppimg and downsampling
        imgc, pxmap = auto_crop(img, dwnsmp_size=20)
        dwnsmp = pxmap[0,1,0] - pxmap[0,0,0]
        origin = pxmap[:,0,0]

        # Fit the cropped image and scale the fit results
        p = GaussianBeam.fit(imgc)
        p['pxc'] = p['pxc']*dwnsmp + origin
        p['x_radius'] *= dwnsmp
        p['y_radius'] *= dwnsmp

        # Fit using the cropped/downsampled pixel map
        p = GaussianBeam.fit(imgc, pxmap=pxmap)


if __name__ == "__main__":
    main()
