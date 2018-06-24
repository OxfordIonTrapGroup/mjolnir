import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from numpy.linalg import eig

from mpl_toolkits.mplot3d import Axes3D

# https://mathematica.stackexchange.com/a/27853
def twod_fit(image):
    # m rows
    # n columns
    m, n = image.shape

    # get a vector of the x,y coordinates of each data point
    x,y = np.mgrid[0:m,0:n]
    x = x.flatten()
    y = y.flatten()
    # this is equivalent to:
    # x = np.outer(np.arange(m), np.ones(n)).flatten()
    # y = np.outer(np.ones(m), np.arange(n)).flatten()

    min_ = np.min(image)
    sum_ = np.sum(image - min_)
    p = ((image - min_)/sum_)

    # phwoar one liner
    #mean = np.array([np.sum(np.multiply(mesh,p))
    #                 for mesh in np.mgrid[0:m,0:n]])

    p = p.flatten()
    mx = np.dot(x, p)
    my = np.dot(y, p)
    mean = np.array([mx, my])
    cov = np.zeros((2,2))
    cov[0,0] = np.dot((x-mx)**2, p)
    cov[0,1] = np.dot((x-mx)*(y-my), p)
    cov[1,0] = cov[0,1]
    cov[1,1] = np.dot((y-my)**2, p)

    w, v = eig(cov)

    return mean, cov, w, v

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    m = 100
    n = 100
    mean = [40, 50]
    cov = [[10, 10],
           [10, 20]]
    w,v = eig(cov)

    # generate image
    # this gives us the location of 10000 clicks
    data = multivariate_normal(mean, cov).rvs(10000)
    # make sure all points are within bounds
    np.clip(data, [0,0], [m-1,n-1])
    data = np.round(data)
    data = data.astype(int)
    #print(data)
    # sum to get image data
    image = np.zeros([m,n],dtype=float)
    for d in data:
        image[d[0],d[1]] += 0.0001

    #print(image)

    # should use mgrid here as well
    # x,y = np.mgrid[0:m,0:n]
    x = np.outer(np.arange(m), np.ones(n))
    y = np.outer(np.ones(m), np.arange(n))
    ax.plot_wireframe(x,y,image)

    mean, cov, w, v = twod_fit(image)

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal(mean,cov)
    ax.plot_surface(x,y,rv.pdf(pos),alpha=0.4)

    print(mean)
    print(w)
    print(v)
    plt.show()


if __name__ == "__main__":
    main()
