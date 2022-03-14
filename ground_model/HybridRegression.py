import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

def make_cov_fcn(x1, x2, l_const=1, sig_f=1, sig_n=1, is_noise=False):
    xx1, xx2 = np.meshgrid(x2, x1) #rows correspond to x1 and columns to x2
    l_const = 2*(l_const**2)
    K = (sig_f**2) * np.exp(-((xx1 - xx2)**2)/l_const)
    if is_noise and len(x1) == len(x2):
        K = K + np.identity(len(x1))*sig_n**2
    return K

#Theta is presumed to be on form: (l_const, sig_f, sig_n)
#seed is ndarray with (alpha, z) elements
#bin_points array of alpha:s
def GPR_predict(seed, bin_points, theta = 0.01*np.array([20, 25, 4])):
    l_const = theta[0]
    sig_f = theta[1]
    sig_n = theta[2]
    z_seed = np.transpose(seed[:, 1])
    cov_1 = make_cov_fcn(bin_points, seed[:, 0], l_const=l_const, sig_f=sig_f, sig_n=sig_n)
    cov_2 = make_cov_fcn(seed[:, 0], seed[:, 0], l_const=l_const, sig_f=sig_f, sig_n=sig_n, is_noise=True)
    z_predict = (cov_1.dot(cov_2)).dot(z_seed)
    return z_predict


# --------- Optimizer functions -----------------------#
def make_partial_grad_matrices(x, theta):
    l_const = theta[0]
    sig_f = theta[1]
    sig_n = theta[2]
    xx1, xx2 = np.meshgrid(x, x)
    d_Kl = (sig_f ** 2) * np.exp(-((xx1 - xx2) ** 2) / (2*l_const**2))*((xx1 - xx2)**2)/(l_const**3)
    d_Ksig_f = 2*sig_f*np.exp(-((xx1 - xx2) ** 2) / (2*l_const**2))
    d_Ksig_n = 2*sig_n*np.identity(len(x))
    return d_Kl, d_Ksig_f, d_Ksig_n

#Takes ndarray with rows (x,y,z) and returns (alpha, z) (ie cartesian to cylindrical without radius)
def convert_coords(input_array):
    alphas = np.arctan2(input_array[:, 1], input_array[:, 0])
    output_array = np.transpose(np.vstack((alphas, input_array[:, 2])))
    return output_array

#data_dict[n] = ndarray of data points (alpha, z) for n:th circle
#Total of N circles [0:N-1]
def hyperparameter_optimizer(data_dict, mean_data_dict, initial_guess):
    print("Hej")
    #Nested function so that the function to be minimized only takes theta as input
    def log_likelihood_loss(theta):
        loss_vect = np.zeros([len(data_dict), ])
        for circle in data_dict: #For every circle(group), calculate loss and add together
            data = data_dict[circle]
            data = convert_coords(data) #to get (alpha, z) - format
            mean = mean_data_dict[circle]
            z = data[:, 1] - mean
            tmp = np.mean(z)
            alpha = data[:, 0]
            cov_matrix = make_cov_fcn(alpha, alpha, theta[0], theta[1], theta[2], is_noise=True)
            cov_inv = np.linalg.inv(cov_matrix)

            term1 = (z.dot(cov_inv)).dot(z)
            term2 = np.log(np.linalg.det(cov_matrix))
            term3 = np.log(2*np.pi)*len(z)
            loss_vect[circle] = 0.5 * (term1 + term2 + term3)

        loss = np.sum(loss_vect)
        return loss

    def loss_gradient(theta):
        grad_vect = np.zeros([len(data_dict), 3])
        for circle in data_dict: #I've chosen the more memory intensive method since it is clearer and likely less computationally heavy - WHAT DID I MEAN?
            data = data_dict[circle]
            data = convert_coords(data) #to get (alpha, z) - format
            mean = mean_data_dict[circle]
            z = data[:, 1] - mean
            tmp = np.mean(z)
            alpha = data[:, 0]
            cov_matrix = make_cov_fcn(alpha, alpha, theta[0], theta[1], theta[2], is_noise=True)
            cov_inv = np.linalg.inv(cov_matrix)
            a_matrix = cov_inv.dot(z)
            common_matrix = np.outer(a_matrix, a_matrix)
            common_matrix = common_matrix - cov_inv

            Kl, Ksigf, Ksign = make_partial_grad_matrices(alpha, theta)

            #Compute hyperparameter-specific components
            grad_l = 0.5*np.trace(common_matrix.dot(Kl))
            grad_sigf = 0.5*np.trace(common_matrix.dot(Ksigf))
            grad_sign = 0.5*np.trace(common_matrix.dot(Ksign))
            grad_vect[circle, :] = -np.array([grad_l, grad_sigf, grad_sign])

        grad = np.sum(grad_vect, axis=0)
        return grad

    result = minimize(log_likelihood_loss, initial_guess, jac=loss_gradient, tol=0.001, method='BFGS')
    print(result)
    #ADD EXCEPTION HANDLING IF BAD INPUT IS GIVEN - RESULT CONTAINS GOOD INFORMATION TO USE
    return result.x


#------------------ Hybrid Regression - main part ---------------------------------------#
#Converts np-array of points into np-array of their bins
def get_bin_keys(points, d_alpha, d_r):
    segs = np.ceil(np.arctan2(points[:, 1], points[:, 0])/d_alpha).astype(int)
    circs = np.ceil(np.linalg.norm(points[:, :2], axis=1)/d_r).astype(int)
    bin_keys = np.transpose(np.array([segs, circs]))
    return bin_keys

#Makes a dictionary with tuples of bin-indices as keys and sorted np-array of points in the bin as value
#Also returns same dictionary with only smallest points
def make_dicts(bin_keys, points):
    bin_indexes = np.unique(bin_keys, axis=0)
    bin_dict = {}
    min_bin_dict = {}
    seg_dict = {}
    min_seg_dict = {}
    circ_dict = {}
    min_circ_dict = {}

    for key in bin_indexes:
        #Returns indexes of points in same bin
        indexes = np.where((bin_keys == key).all(axis=1))[0]
        points_in_bin = points[indexes, :]
        #Sort these points according to z-axis
        points_in_bin = points_in_bin[np.argsort(points_in_bin[:, 2])]
        bin_dict[tuple(key)] = points_in_bin
        min_bin_dict[tuple(key)] = points_in_bin[0, :]

        #Same thing but for the different segments
        indexes_segs = np.where(bin_keys[:, 0] == key[0])[0]
        points_in_seg = points[indexes_segs, :]
        points_in_seg = points_in_seg[np.argsort(points_in_seg[:, 2])]
        seg_dict[key[0]] = points_in_seg
        min_seg_dict[key[0]] = points_in_seg[0, :]

        #Same thing but for different circles
        indexes_circs = np.where(bin_keys[:, 1] == key[1])[0]
        points_in_circ = points[indexes_circs, :]
        points_in_circ = points_in_circ[np.argsort(points_in_circ[:, 2])]
        circ_dict[key[1]] = points_in_circ
        min_circ_dict[key[1]] = points_in_circ[0, :]

    return bin_dict, min_bin_dict, seg_dict, min_seg_dict, circ_dict, min_circ_dict


def find_min_max_index(bin_keys):
    segs = bin_keys[:, 0]
    circs = bin_keys[:, 1]
    max_min_segment = [np.min(segs), np.max(segs)]
    max_depth_ind = np.max(circs)

    return max_depth_ind, max_min_segment

#Takes in min_bin_dict and for every bin in a segment, retreives the smallest values (min_bin_dict)
#Prodces a dictionary of arrays for every segment
# If a bin between 0 and max_depth_index has no value, this is skipped. all circles - [0:max_depth_index]
def make_seg_line(min_bin_dict, max_depth_index, min_max_segment):
    SLm_dict = {} #SLm_dict[m] gives array of points for the m:th segment
    circle_index_dict = {} #complementary dictionary to save which circle points belong to in a segment
    for segment in range(min_max_segment[0], min_max_segment[1]+1):
        segment_array = np.empty((0, 3), float)
        circle_array = np.empty((0, 1), int)
        for circle in range(max_depth_index+1):
            if (segment, circle) in min_bin_dict:
                tmp_array = min_bin_dict[segment, circle]
                segment_array = np.vstack([segment_array, tmp_array])
                circle_array = np.vstack([circle_array, np.array([circle])])
        SLm_dict[segment] = segment_array
        circle_index_dict[segment] = circle_array
    return SLm_dict, circle_index_dict

def make_circles(seg_dict, circle_dict_index, max_depth_index, min_max_segment):
    circle_dict = {}
    first_segment = True
    for segment in range(min_max_segment[0], min_max_segment[1]+1):
        segment_array = seg_dict[segment]
        circle_index_array = circle_dict_index[segment]
        for circle_index in range(max_depth_index+1):
            if first_segment: #loop of first segment has to create empty arrays for stacking
                circle_array = np.empty((0, 3), float)
            else:
                circle_array = circle_dict[circle_index]
            inds = np.argwhere(circle_index_array[:, 0] == circle_index)
            inds = inds.reshape(inds.size)
            tmp_array = segment_array[inds, :]
            #print(str(circle_index == np.ceil(np.linalg.norm(tmp_array[:, :2], axis=1) / 1)) + " " + str(circle_index))
            circle_dict[circle_index] = np.vstack([circle_array, tmp_array])
        first_segment = False

    return circle_dict

def circle_mean(circle_dict):
    mean_circle_dict = {}
    for circle in circle_dict:
        circle_array = circle_dict[circle]
        mean_circle_dict[circle] = np.mean(circle_array[:, 2])
    return mean_circle_dict
    #for bin in range()

#assumes 1-dim input in pos
def grad_filter(value, pos, large_slope):
    gradients = np.gradient(value, pos)  # - regr the f-values and r is the spacing
    degrees = np.arctan(gradients) * (180 / np.pi) #conversion to degrees for comparison
    large_slope_indexes = np.argwhere(degrees >= large_slope)
    large_slope_indexes = large_slope_indexes.reshape(large_slope_indexes.shape[0],)
    clean_position_array = np.delete(pos, large_slope_indexes)
    clean_value_array = np.delete(value, large_slope_indexes)

    #For every index in large_slope_indexes, find closest pos in clean_position_array
    for ind in large_slope_indexes:
        if len(clean_position_array) == 0:
            print("Warning: All gradients in segment above large slope threshold")
        else:
            idx = (np.abs(clean_position_array - pos[ind])).argmin() #index in clean array which is to be replicated
            value[ind] = clean_value_array[idx]

    return value

#Take in a dictionary of segment ndarrays, filters these with the RLWR-regression and smoothes the result with a gradient filter
#Will be returned in the same format as given
def filter_segments(SLm_dict, fract, iterations, large_slope):
    lowess = sm.nonparametric.lowess
    for segment in SLm_dict:
        segment_points = SLm_dict[segment]
        z = segment_points[:, 2]
        if len(z) < 2:
            print("Warning: segment contains only one value")
            filtered_values = z #If we cannot use grad filter, filtered values are assigned the intitial values

        else:
            r = np.linalg.norm(segment_points[:, :2], axis=1)
            regr = lowess(z, r, frac=fract, it=iterations, return_sorted=False)

            # Gradient filter on points
            filtered_values = grad_filter(regr, r, large_slope)

        segment_points[:, 2] = filtered_values
        SLm_dict[segment] = segment_points
    return SLm_dict
"""
        print(segment*0.1)
        #Plot the segments
        fig = plt.figure(0)
        #ax = fig.add_subplot(projection='2d')
        plt.scatter(r, z)
        plt.show()
"""

#Takes in average bin height and points in bin:
#Outputs array of indexes for filtered points
def filter_points_hybrid(bin_average, heights, threshold):
    ground_indeces = np.argwhere(np.abs(bin_average - heights) < threshold)
    return ground_indeces.reshape(len(ground_indeces),)

#Fill out gaps in average_bin_dict
def fill_out_average_bins(bin_average_dict, max_depth_index, min_max_segment, circle_mean_dict, circle_dict, d_alpha, theta):
    for circle in range(max_depth_index):  # Should I include zero?
        mean = circle_mean_dict[circle]

        #Extract seed points for circle and points in the given bin
        seed_points_cart = circle_dict[circle]
        seed_points_cylindrical = convert_coords(seed_points_cart)
        seed_points_cylindrical[:, 1] = seed_points_cylindrical[:, 1] - mean

        for segment in range(min_max_segment[0], min_max_segment[1] + 1):
            if (segment, circle) not in bin_average_dict:
                #Centre point of bin
                # Go from segment to centre of segment angle alpha
                alpha = (segment-0.5)*d_alpha
                bin_average_dict[(segment, circle)] = GPR_predict(seed_points_cylindrical, alpha, theta)

#def GPR_predict(seed, bin_points, theta = 0.01*np.array([20, 25, 4])):

    return bin_average_dict

def GPR_predicter(circle_dict, bin_dict, circle_mean_dict, threshold, max_depth_index, min_max_segment, d_alpha, theta=0.01*np.array([20, 25, 4])):
    bin_average_dict = {}
    filtered_points_in_bin_dict = {}
    for bin_tuple in bin_dict:
        segment = bin_tuple[0]
        circle = bin_tuple[1]
        mean = circle_mean_dict[circle]

        #Extract seed points for circle and points in the given bin
        seed_points_cart = circle_dict[circle]
        seed_points_cylindrical = convert_coords(seed_points_cart)
        bin_points_cart = bin_dict[bin_tuple]
        bin_points_cylindrical = convert_coords(bin_points_cart)

        #Make zero mean for passing into GPR
        seed_points_cylindrical[:, 1] = seed_points_cylindrical[:, 1] - mean
        bin_points_cylindrical[:, 1] = bin_points_cylindrical[:, 1] - mean

        #Prediction
        heights = GPR_predict(seed_points_cylindrical, bin_points_cylindrical[:, 0], theta=theta)  # predicts on one bin at the time
        true_predicted_heights = heights + mean #Mean corrected heights of GPR-prediction
        average_height = np.mean(true_predicted_heights)
        bin_average_dict[bin_tuple] = average_height

        filtered_points_indexes = filter_points_hybrid(average_height, bin_points_cart[:, 2], threshold)
        filtered_heights = bin_points_cart[filtered_points_indexes, 2]

        filtered_points = np.hstack([bin_points_cart[filtered_points_indexes, :2], filtered_heights.reshape(len(filtered_heights), 1)])
        filtered_points_in_bin_dict[bin_tuple] = filtered_points

        bin_points_cart[:, 2] = true_predicted_heights
        bin_dict[bin_tuple] = bin_points_cart

    #FILL OUT bin_average_dict TO INVOLVE ALL INCLUDED BIN THAT HAVEN'T BEEN COVERED
    #HERE
    bin_average_dict = fill_out_average_bins(bin_average_dict, max_depth_index, min_max_segment, circle_mean_dict, circle_dict, d_alpha, theta)

    return bin_dict, bin_average_dict, filtered_points_in_bin_dict

def bin_dict2pcl(bin_dict):
    pcl_array = np.empty((0, 3), float)
    for bin in bin_dict:
        pcl_array = np.vstack([pcl_array, bin_dict[bin]])

    return pcl_array

#takes the prediction for every bin and returns function from (x,y) to representative bin's predicted height z_hat
def make_ground_function(bin_average_dict, d_r, d_alpha):

    #points ndarray on form (x,y)
    def ground_func(points): #SHOULD BE POSSIBLE TO PARALELLIZE WITH #np.vectorize(dict.get)(x) AND SO ON - JUST NEED TO CONVERT NDARRAY TO NDARRAY OF TUPLES
        keys = get_bin_keys(points, d_alpha, d_r) #Nx2 ndarray
        heights = np.zeros(shape=(points.shape[0], 1))
        for row in range(points.shape[0]):
            key = keys[row, :]
            if (key[0], key[1]) not in bin_average_dict:
                print("point with index " + str(row) + " is outside ground area!")
                heights[row] = None
            else:
                heights[row] = bin_average_dict[(key[0], key[1])]

        return heights

    return ground_func

# point_cloud, dx, dy, fract = 1/20, threshold = 0.1, delta = 0.2, iterations = 2, do_print = False
def hybrid_regression(point_cloud, d_alpha=(np.pi/2)/2, d_r=1, fract=1/20, threshold=0.3, iterations=5, large_slope=10, max_range=100):

    #Filter points
    indexes = np.argwhere(np.linalg.norm(point_cloud[:, :2], axis=1) < max_range)
    point_cloud = point_cloud[indexes.reshape(len(indexes),), :]


    bin_keys = get_bin_keys(point_cloud, d_alpha, d_r)
    bin_dict, min_bin_dict, seg_dict, min_seg_dict, circ_dict, min_circ_dict = make_dicts(bin_keys, point_cloud)
    max_depth_ind, min_max_segment = find_min_max_index(bin_keys)

    #RWLR-step
    SLm, circle_index_dictionary = make_seg_line(min_bin_dict, max_depth_ind, min_max_segment)
    sdm = filter_segments(SLm, fract=fract, iterations=iterations, large_slope=large_slope)

    #GPR-step
    SLn = make_circles(sdm, circle_index_dictionary, max_depth_ind, min_max_segment)
    circle_mean_dict = circle_mean(SLn)

    #From other method
    theta = np.array([-0.25473245, 0.3997241, 0.28976028]) # - Pretty bad
    theta = 0.01 * np.array([20, 25, 4])
    ########################
    bin_dict_predictions, bin_average_dict, filtered_points_dict = GPR_predicter(SLn, bin_dict, circle_mean_dict,
                                                                                 threshold, theta=theta, max_depth_index=max_depth_ind, min_max_segment=min_max_segment, d_alpha=d_alpha)
    #max_depth, min_max_segment, d_alpha
    # bin_dict_final = filter_points_hybrid(bin_average_dict, bin_dict, threshold)
    point_cloud_fit = bin_dict2pcl(bin_dict_predictions)
    ground_points = bin_dict2pcl(filtered_points_dict)

    ground_function = make_ground_function(bin_average_dict, d_r, d_alpha)


    #TODO: Add optimizer step
    return ground_points,  ground_function, point_cloud_fit


#------------------ Computations to be added in main ------------------------------------#
if False:
    a = np.array([1])
    b = np.array([1, 2, 3])
    #print(make_cov_fcn(a, b))
    #print(make_cov_fcn(b,b,is_noise=True))
    s = np.array([[0.1, 1],
                  [0.2, 0.4],
                  [0.3, -0.4],
                  [0.4, -1]])
    bin_points = np.array([0.6])
    print(GPR_predict(s, bin_points))

if False:
    def test_func(x):
        return (x[0] ** 2) * (x[1] ** 2)

    def test_grad(x):
        return np.array([2 * x[0] * x[1] ** 2, 2 * (x[1] ** 2) * x[0]])


    init = np.array([2, 2])
    res = minimize(fun=test_func, x0=init, jac=test_grad, tol=0.01, method='BFGS')
    print(res.x)

if False: #test partial grad-matrices
    x = np.linspace(0,1,3)
    print(x)
    theta = np.array([1,1,1])
    Kl, Ksf, Ksn = make_partial_grad_matrices(x, theta)
    print(Kl)
    print(Ksf)
    print(Ksn) #Seems correct

if False:
    data = {}
    data[0] = np.array([[0.1, 1],
                  [0.2, 0.4],
                  [0.3, -0.3],
                  [0.4, -1]])
    data[1] = np.array([[0.11, 1],
                  [0.22, 0.3],
                  [0.31, -0.6],
                  [0.394, -1]])

    data_dict = data
    """
    def log_likelihood_loss(theta):
        loss_vect = np.zeros([len(data_dict), ])
        for group in range(len(data_dict)): #For every circle(group), calculate loss and add together
            data = data_dict[group]
            z = data[:, 1]
            alpha = data[:, 0]
            cov_matrix = make_cov_fcn(alpha, alpha, theta[0], theta[1], theta[2], is_noise=True)
            loss_vect[group] = 0.5 * ((z.dot(cov_matrix)).dot(z) + np.log(np.linalg.det(cov_matrix)) + np.log(2*np.pi)*len(z))

        return np.sum(loss_vect)
        
    init = np.array([1, 1, 1])
    print(log_likelihood_loss(init))
    """

    """
    def loss_gradient(theta):
        grad_vect = np.zeros([len(data_dict), 3])
        for group in range(len(data_dict)):  # I've chosen the more memory intensive method since it is clearer and likely less computationally heavy
            data = data_dict[group]
            z = data[:, 1]
            alpha = data[:, 0]
            cov_matrix = make_cov_fcn(alpha, alpha, theta[0], theta[1], theta[2], is_noise=True)
            cov_inv = np.linalg.inv(cov_matrix)
            a_matrix = cov_inv.dot(z)
            common_matrix = a_matrix.dot(a_matrix) - cov_inv

            Kl, Ksigf, Ksign = make_partial_grad_matrices(alpha, theta)

            # Compute hyperparameter-specific components
            grad_l = 0.5 * np.trace(common_matrix.dot(Kl))
            grad_sigf = 0.5 * np.trace(common_matrix.dot(Ksigf))
            grad_sign = 0.5 * np.trace(common_matrix.dot(Ksign))
            grad_vect[group, :] = np.array([-grad_l, -grad_sigf, -grad_sign])

        return np.sum(grad_vect, axis=0)
    init = np.array([1, 1, 1])
    print(loss_gradient(init))"""

    init = 0.01*np.array([20, 25, 4])
    print(hyperparameter_optimizer(data, init))

if False:
    # point_cloud, dx, dy, fract = 1/20, threshold = 0.1, delta = 0.2, iterations = 2, do_print = False
    def hybrid_regression_tmp(point_cloud, d_alpha=(np.pi / 2) / 2, d_r=1, fract=1 / 20, threshold=0.3, iterations=5,
                          large_slope=10):
        threshold = threshold
        d_alpha = d_alpha
        d_r = d_r
        bin_keys = get_bin_keys(point_cloud, d_alpha, d_r)
        bin_dict, min_bin_dict, seg_dict, min_seg_dict, circ_dict, min_circ_dict = make_dicts(bin_keys, point_cloud)
        max_depth_ind, min_max_segment = find_min_max_index(bin_keys)

        # RWLR-step
        SLm, circle_index_dictionary = make_seg_line(min_bin_dict, max_depth_ind, min_max_segment)
        sdm = filter_segments(SLm, fract=fract, iterations=iterations, large_slope=large_slope)

        # GPR-step
        SLn = make_circles(sdm, circle_index_dictionary, max_depth_ind, min_max_segment)
        circle_mean_dict = circle_mean(SLn)

        #-----------ADDED OPTIMIZER STEP -----------------------#
        #data_dict: seeds for every circle
        #N = max_depth_ind
        theta = hyperparameter_optimizer(data_dict=SLn, mean_data_dict=circle_mean_dict, initial_guess=0.01*np.array([20, 25, 4]))

        #---------------------------------------------------------#
        bin_dict_predictions, bin_average_dict, filtered_points_dict = GPR_predicter(SLn, bin_dict, circle_mean_dict,
                                                                                     threshold, theta=theta)
        # bin_dict_final = filter_points_hybrid(bin_average_dict, bin_dict, threshold)
        point_cloud_fit = bin_dict2pcl(bin_dict_predictions)
        ground_points = bin_dict2pcl(filtered_points_dict)

        # TODO: Add optimizer step
        return ground_points, bin_average_dict, point_cloud_fit

    #DATA FOR RUNNING ABOVE
    n = 100
    x = np.linspace(0.01, 4.99, n)
    y = np.linspace(0.01, 4.99, n)
    xx, yy = np.meshgrid(x, y)
    degf = 3

    noise = ((np.random.chisquare(df=degf, size=(n, n)) - (
            degf - 2)) / 5)  # ** 2  # The method works a lot better with noise that is chisquare**2
    zz = np.multiply(np.sin(xx), np.cos(yy)) + noise

    nrows, ncols = xx.shape
    xarray = np.reshape(xx, (nrows * ncols, 1))[:, 0]
    yarray = np.reshape(yy, (nrows * ncols, 1))[:, 0]
    zarray = np.reshape(zz, (nrows * ncols, 1))[:, 0]

    pcl = np.transpose(np.array([xarray, yarray, zarray]))

    # Hybrid Regression step
    gp, bad, pcf = hybrid_regression_tmp(pcl, d_r=0.1, d_alpha=0.02, threshold=0.1)