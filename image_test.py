import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from numpy.linalg import eig
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from image_tools import twod_fit


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
    ax.plot_wireframe(x,y,image)

    r = twod_fit(image)

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal(r['mean'],r['cov'])
    ax.plot_surface(x,y,(rv.pdf(pos)*r['scale'])+r['min'],alpha=0.4)

    plt.show()


if __name__ == "__main__":
    main()
