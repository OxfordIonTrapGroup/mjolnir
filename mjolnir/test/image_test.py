import numpy as np
from scipy.stats import multivariate_normal as mv
# import matplotlib.pyplot as plt
# from matplotlib import cm
from numpy.linalg import eig
# import matplotlib.image as mpimg
# from mpl_toolkits.mplot3d import Axes3D

from mjolnir.tools.image_tools import gaussian_beam


def generate_image(
        centroid=[200, 200],
        cov=[[50,0],[0,80]],
        intensity=220,
        noise=None,
        m=1280,
        n=1024):

    pxmap = np.mgrid[0:m,0:n]
    pxmap = np.einsum("i...->...i", pxmap)
    img = mv(centroid, cov).pdf(pxmap)

    # scale it
    img *= intensity/np.amax(img)

    # TODO: add noise
    if noise is not None:
        pass

    return img.astype(int)



def load_image(fname):
    img = mpimg.imread(fname)
    m, n = img.shape
    img = img.astype(float)
    return m, n, img


def calculate_residuals(ydata, yfit):
    return np.sum(np.abs(ydata-yfit))

def calculate_quality(ydata, yfit):
    rnorm = calculate_residuals(ydata, yfit)
    s = np.sum(ydata)
    return 1. - (rnorm/s)

def gaussian_content(ydata, yfit):
    """experimental figure of merit"""
    # actually this isn't great, doesn't penalise elliptical beams
    total = np.sum(np.multiply(ydata, yfit))
    norm = np.sum(np.multiply(ydata, ydata))
    return total/norm

def main():
    # Plot a 3d mesh of the image data
    # and a surface of the fit
    # maybe use this to test eigenvectors as well
    fig = plt.figure()

    n, m, image = load_image("focus.bmp")
    image = np.transpose(image)

    x = np.mgrid[0:m,0:n]
    pos = np.einsum("i...->...i", x)

    methods = [
        gaussian_beam.naive_fit,
        # gaussian_beam.two_step_fit,
        gaussian_beam.two_step_fit_mk2,
        # gaussian_beam.lsq_fit,
        gaussian_beam.lsq_cropped
    ]
    fits = [meth(x, image) for meth in methods]
    fit_points = [mv(f['x0'], f['cov']).pdf(pos)*f['scale']+f['offset']
                  for f in fits]

    centx, centy = fits[0]['x0'].astype(int)
    hr = 25 #halfsize of plot region
    pos = pos[centx-hr:centx+hr, centy-hr:centy+hr,:]
    ydata = image[centx-hr:centx+hr, centy-hr:centy+hr]
    print("sum: ", np.sum(ydata))

    nplots = len(methods)
    for i in range(nplots):
        ax = fig.add_subplot(nplots, 1, i+1, projection='3d')
        ax.plot_wireframe(pos[:,:,0], pos[:,:,1], ydata)

        yfit = fit_points[i][centx-hr:centx+hr, centy-hr:centy+hr]
        ax.plot_surface(pos[:,:,0], pos[:,:,1], yfit,
            cmap=cm.coolwarm, alpha=0.6)

        res = calculate_residuals(ydata, yfit)
        q = calculate_quality(ydata, yfit)
        gc = gaussian_content(ydata, yfit)
        print("res norm: ", res)
        print("quality: ", q)
        print("gc: ", gc)

    plt.show()


if __name__ == "__main__":
    main()
