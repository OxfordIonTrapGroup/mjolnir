import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import eig
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from image_tools import twod_fit, fit_image
from gaussian_beam import gaussian_beam


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

    m, n, image = load_image("focus.bmp")

    x,y = np.mgrid[0:m,0:n]

    prelim = twod_fit(image)
    xmean,ymean = np.around(prelim['mean']).astype(int)
    mask = np.full_like(image, False, dtype=bool)
    mask[xmean-25:xmean+25, ymean-25:ymean+25] = True

    doctored = np.copy(image)
    doctored[~mask] = 0.
    r = twod_fit(doctored)
    o = fit_image(image)

    xy = np.mgrid[0:m,0:n]
    # p = gaussian_beam().fit(xy, image)

    # print(r['mean'], p['mean'])
    # print(r['cov'], p['cov'])
    # print(r['scale'], p['scale'])

    pos = np.empty((50,50,2))

    pos[:, :, 0] = x[xmean-25:xmean+25, ymean-25:ymean+25]
    pos[:, :, 1] = y[xmean-25:xmean+25, ymean-25:ymean+25]

    ax.plot_wireframe(pos[:,:,0], pos[:,:,1],
        image[xmean-25:xmean+25, ymean-25:ymean+25])

    rv = multivariate_normal(r['mean'],r['cov'])
    ax.plot_surface(pos[:,:,0], pos[:,:,1], (rv.pdf(pos)*r['scale'])+r['min'],
        cmap=cm.coolwarm, alpha=0.6)

    # rv = multivariate_normal(p['mean'],p['cov'])
    # ax.plot_surface(pos[:,:,0], pos[:,:,1], (rv.pdf(pos)*p['scale'])+p['offset'],
    #     cmap=cm.plasma, alpha=0.6)


    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(r['row_x_data'], r['row_y_data'], marker='x', alpha=0.6)
    ax.plot(r['row_x_fit'],r['row_y_fit'])
    ax.scatter(o['row_x_data'], o['row_y_data'], alpha=0.6)
    ax.plot(o['row_x_fit'],o['row_y_fit'])

    ax = fig.add_subplot(212)
    ax.scatter(r['col_x_data'], r['col_y_data'], marker='x', alpha=0.6)
    ax.plot(r['col_x_fit'],r['col_y_fit'])
    ax.scatter(o['col_x_data'], o['col_y_data'], alpha=0.6)
    ax.plot(o['col_x_fit'],o['col_y_fit'])

    plt.show()


if __name__ == "__main__":
    main()
