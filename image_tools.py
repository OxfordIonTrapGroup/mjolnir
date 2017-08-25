
# import matplotlib.image as mpimage
import numpy as np
from oitg.fitting.gaussian_beam import gaussian_beam

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

    p, p_error, x_fit, y_fit = gaussian_beam.fit(
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

# if __name__ == "__main__":

#     file_name = 'focus.bmp'
#     (width, height), m = load_image(file_name)

#     print(find_center(m))
