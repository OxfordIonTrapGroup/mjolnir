
# import matplotlib.image as mpimage
import numpy as np
from scipy.stats import multivariate_normal
from oitg.fitting.gaussian import gaussian

import scipy.ndimage.filters

pixel_size = 5.2

# def load_image(filename):
#     image = mpimage.imread(filename)
#     height, width = image.shape[0:2]

#     # If the image is not mono, make it mono
#     if len(image.shape) > 2:
#         # not implemented
#         raise Exception('Conversion to mono not yet implemented')
#     else:
#         mono_image = image.astype(float)

#     # Find the sum of all the pixels
#     pixel_sum = np.sum(mono_image)

#     return (width, height), mono_image

def fit_slice(x_data, y_data, x0):

    p, p_error, x_fit, y_fit = gaussian.fit(
        x_data, y_data, evaluate_function=True,
        initialise={'x0':x0})

    return p, p_error, x_fit, y_fit

def find_center(image):
    sigma = 10
    filtered_image = scipy.ndimage.filters.gaussian_filter(image, sigma)

    a, b = np.unravel_index(filtered_image.argmax(), filtered_image.shape)

    row_y_data = filtered_image[a,:]
    row_x_data = np.arange(len(row_y_data))

    row_p, row_p_error, row_x_fit, row_y_fit = fit_slice(
        row_x_data, row_y_data, b)

    col_y_data = filtered_image[:,b]
    col_x_data = np.arange(len(col_y_data))

    col_p, col_p_error, col_x_fit, col_y_fit = fit_slice(
        col_x_data, col_y_data, a)

    x = row_p['x0']
    wx = row_p['w0']
    y = col_p['x0']
    wy = col_p['w0']

    return x, y, wx, wy

def fit_image(image):
    # Size of mask in units of w0
    mask_size = 1

    # Make an initial guess
    x, y, wx, wy = find_center(image)

    mask_lim_x = np.array([x - mask_size*wx, x + mask_size*wx])
    mask_lim_x = np.around(mask_lim_x).astype(int)
    mask_lim_x = np.maximum(mask_lim_x,[0,0])
    mask_lim_x = np.minimum(mask_lim_x,len(image[0,:]))
    mask_x = np.arange(mask_lim_x[0], mask_lim_x[1])

    mask_lim_y = np.array([y - mask_size*wy, y + mask_size*wy])
    mask_lim_y = np.around(mask_lim_y).astype(int)
    mask_lim_y = np.maximum(mask_lim_y,[0,0])
    mask_lim_y = np.minimum(mask_lim_y,len(image[:,0]))
    mask_y = np.arange(mask_lim_y[0], mask_lim_y[1])


    row_y_data = image[int(y),mask_lim_x[0]:mask_lim_x[1]]
    row_x_data = mask_x

    row_p, row_p_error, row_x_fit, row_y_fit = fit_slice(row_x_data, row_y_data, x)

    col_y_data = image[mask_lim_y[0]:mask_lim_y[1], int(x)]
    col_x_data = mask_y

    col_p, col_p_error, col_x_fit, col_y_fit = fit_slice(col_x_data, col_y_data, y)

    x = row_p['x0']
    wx = row_p['w0']
    y = col_p['x0']
    wy = col_p['w0']

    amp_x = row_p['a']
    amp_y = col_p['a']

    outlined_image = np.zeros(image.shape)
    lw = 20
    val = 255
    outlined_image[mask_lim_y[0]-int(lw/2):mask_lim_y[1]+int(lw/2),
                mask_lim_x[0]-int(lw/2):mask_lim_x[0]+int(lw/2)] = val
    outlined_image[mask_lim_y[0]-int(lw/2):mask_lim_y[1]+int(lw/2),
                mask_lim_x[1]-int(lw/2):mask_lim_x[1]+int(lw/2)] = val
    outlined_image[mask_lim_y[0]-int(lw/2):mask_lim_y[0]+int(lw/2),
                mask_lim_x[0]-int(lw/2):mask_lim_x[1]+int(lw/2)] = val
    outlined_image[mask_lim_y[1]-int(lw/2):mask_lim_y[1]+int(lw/2),
                mask_lim_x[0]-int(lw/2):mask_lim_x[1]+int(lw/2)] = val

    fit_results = {}

    fit_results["image_zoom"] = image[mask_lim_y[0]:mask_lim_y[1],\
        mask_lim_x[0]:mask_lim_x[1]]

    fit_results["outlined_image"] = outlined_image

    fit_results["row_x_fit"] = row_x_fit
    fit_results["row_y_fit"] = row_y_fit
    fit_results["row_x_data"] = row_x_data
    fit_results["row_y_data"] = row_y_data

    fit_results["col_y_fit"] = col_y_fit
    fit_results["col_x_fit"] = col_x_fit
    fit_results["col_y_data"] = col_y_data
    fit_results["col_x_data"] = col_x_data

    fit_results["wx"] = wx*pixel_size
    fit_results["wy"] = wy*pixel_size

    fit_results["x"] = x*pixel_size
    fit_results["y"] = y*pixel_size

    fit_results["amp_x"] = amp_x
    fit_results["amp_y"] = amp_y

    return fit_results


# note that this code may have x and y swapped, please check!
#
# algorithm from:
# https://mathematica.stackexchange.com/a/27853
m = 1024
n = 1280
# gets a matrix each for the x,y coordinates of each data point
xx,yy = np.mgrid[0:m,0:n]
x = xx.flatten()
y = yy.flatten()
def twod_fit(image):
    # m columns
    # n rows
    # m, n = image.shape

    # rescale so that this is a probability distribution
    # makes following calculations easier
    min_ = np.amin(image)
    sum_ = np.sum(image - min_)
    p = ((image - min_)/sum_).flatten()

    mx = np.dot(x, p)
    my = np.dot(y, p)
    mean = np.array([mx, my])

    cov = np.zeros((2,2))
    cov[0,0] = np.dot((x-mx)**2, p)
    cov[0,1] = np.dot((x-mx)*(y-my), p)
    cov[1,0] = cov[0,1]
    cov[1,1] = np.dot((y-my)**2, p)

    e_val, e_vec = np.linalg.eig(cov)

    results = {}
    results['mean'] = mean    # vector containing centre x and y pixel values
    results['cov'] = cov      # covariance matrix
    results['min'] = min_
    results['scale'] = sum_
    results['pixel_size'] = pixel_size

    # eigenvalues are the variance in the direction of the eigenvectors
    results['eigvals'], results['eigvecs'] = np.linalg.eig(cov)

    # back compatible return values
    derived = derived_results(image, mean, cov, min_, sum_)
    results.update(derived)

    return results


def derived_results(image, mean, cov, offset, scale):
    # ======================================
    # now backwards compatible return values
    #
    def stddev_xy(cov):
        """Return stddev radius in x,y from covariance matrix"""
        # think this is right but not 100% sure
        # we're inverting the covariance matrix
        # see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Estimation_of_parameters
        varx = np.abs(cov[0,0] - cov[0,1]*cov[1,0])
        vary = np.abs(cov[1,1] - cov[0,1]*cov[1,0])
        # shouldn't need the abs...
        var = np.array([varx, vary])

        return np.sqrt(var)

    def amplitude(cov, scale, offset=0):
        """Return peak value of 2D gaussian from covariance matrix,
        pizel sum and minimum"""
        return offset + scale/(2*np.pi*np.sqrt(np.linalg.det(cov)))

    def get_mask_lim(centre, size, max_pixel):
        lim = np.array([centre - size/2, centre + size/2])
        lim = np.around(lim).astype(int)
        lim = np.clip(lim, 0, max_pixel)
        return lim

    results = {}
    # uses simple list unpacking
    results['x'], results['y'] = mean*pixel_size

    # width in pixels (still floating point)
    wp = stddev_xy(cov)
    results['wx'], results['wy'] = wp*pixel_size

    # these should be the same by definition: now they are
    amp = amplitude(cov, scale, offset)
    results["amp_x"] = amp
    results["amp_y"] = amp

    mask_size = 0.5
    limx = get_mask_lim(mean[0], mask_size*wp[0], n)
    limy = get_mask_lim(mean[1], mask_size*wp[1], m)

    outlined_image = np.zeros(image.shape)
    hlw = 10  # half width of line in pixels
    val = 255 # pixel value
    outlined_image[limy[0]-hlw:limy[1]+hlw, limx[0]-hlw:limx[0]+hlw] = val
    outlined_image[limy[0]-hlw:limy[1]+hlw, limx[1]-hlw:limx[1]+hlw] = val
    outlined_image[limy[0]-hlw:limy[0]+hlw, limx[0]-hlw:limx[1]+hlw] = val
    outlined_image[limy[1]-hlw:limy[1]+hlw, limx[0]-hlw:limx[1]+hlw] = val

    mv = multivariate_normal(mean, cov)
    x0, y0 = mean.astype(int)

    # NB confusing notation here (to match up to Tim's existing code)
    # xx is the matrix of x positions of each pixel
    # (where x is the *column* number)
    # yy is similar for y and row number
    # row_x_data/row_x_fit are both just the pixel positions in the row
    # row_y_data/row_y_fit are the pixel (real/fitted) values
    # pos gives the pixel (x,y) positions, needed for 2d normal fit values
    #
    pos = np.empty((limy[1]-limy[0], 2))
    pos[:,0] = xx[x0,limy[0]:limy[1]]
    pos[:,1] = yy[x0,limy[0]:limy[1]]
    results["row_x_fit"] = pos[:,1]
    results["row_y_fit"] = (mv.pdf(pos)*scale)+offset
    results["row_x_data"] = pos[:,1]
    results["row_y_data"] = image[x0,limy[0]:limy[1]]

    pos = np.empty((limx[1]-limx[0], 2))
    pos[:,0] = xx[limx[0]:limx[1],y0]
    pos[:,1] = yy[limx[0]:limx[1],y0]
    results["col_x_fit"] = pos[:,0]
    results["col_y_fit"] = (mv.pdf(pos)*scale)+offset
    results["col_x_data"] = pos[:,0]
    results["col_y_data"] = image[limx[0]:limx[1],y0]

    results["image_zoom"] = image[limy[0]:limy[1],limx[0]:limx[1]]
    results["outlined_image"] = outlined_image

    #  this is for trying a multivariate lsq
    results["limx"] = limx
    results["limy"] = limy

    return results

# if __name__ == "__main__":

#     file_name = 'focus.bmp'
#     (width, height), m = load_image(file_name)

#     print(find_center(m))
