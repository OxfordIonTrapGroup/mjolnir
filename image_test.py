import numpy as np
from scipy.stats import multivariate_normal as mv
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import eig
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from new_image_tools import gaussian_beam


def generate_image():
    m = 1280
    n = 1024
    mean = [600, 500]
    cov = [[100, 100],
           [100, 200]]

    # generate image
    # this gives us the location of nclicks
    nclicks = 1000
    data = multivariate_normal(mean, cov).rvs(nclicks)
    # make sure all points are within bounds
    # this isn't perfect but so long as not too many points are near
    # the edge then it's fine
    np.clip(data, [0,0], [m-1,n-1])
    data = np.round(data)
    data = data.astype(int)

    # sum to get image data
    img = np.zeros([m,n],dtype=float)
    for d in data:
        img[d[0],d[1]] += 1.0/nclicks

    # scale it
    max_ = 220.0
    img *= max_/np.maximum(img)

    return m, n, img



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
