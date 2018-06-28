import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import eig
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from image_tools import twod_fit, fit_image


def generate_image(m,n):
    m = 1024
    n = 1280
    mean = [400, 500]
    cov = [[100, 100],
           [100, 200]]

    # generate image
    # this gives us the location of nclicks
    nclicks
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
    max_ = 225.0
    img *= max_/np.maximum(img)

    return m, n, img



def load_image(fname):
    img = mpimg.imread(fname)
    m, n = img.shape
    img = img.astype(float)
    return m, n, img


def main():
    # Plot a 3d mesh of the image data
    # and a surface of the fit
    # maybe use this to test eigenvectors as well
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    m, n, image = load_image("colimated.bmp")

    x,y = np.mgrid[0:m,0:n]

    surf = ax.plot_surface(x, y, image, cmap=cm.gray, alpha=0.2)

    r = twod_fit(image)
    o = fit_image(image)

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal(r['mean'],r['cov'])
    ax.plot_surface(x, y, (rv.pdf(pos)*r['scale'])+r['min'],
        cmap=cm.coolwarm, alpha=0.6)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(r['row_x_data'], r['row_y_data'])
    ax.plot(r['row_x_fit'],r['row_y_fit'])
    ax.scatter(o['row_x_data'], o['row_y_data'])
    ax.plot(o['row_x_fit'],o['row_y_fit'])

    ax = fig.add_subplot(212)
    ax.scatter(r['col_x_data'], r['col_y_data'])
    ax.plot(r['col_x_fit'],r['col_y_fit'])
    ax.scatter(o['col_x_data'], o['col_y_data'])
    ax.plot(o['col_x_fit'],o['col_y_fit'])

    plt.show()


if __name__ == "__main__":
    main()
