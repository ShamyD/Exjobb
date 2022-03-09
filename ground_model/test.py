import numpy as np
#import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from HybridRegression import *

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
        r = np.linalg.norm(segment_points[:, :2], axis=1)
        regr = lowess(z, r, frac=fract, it=iterations, return_sorted=False)
        #Gradient filter on points
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


#Takes ndarray with rows (x,y,z) and returns (alpha, z) (ie cartesian to cylindrical without radius)
def convert_coords(input_array):
    alphas = np.arctan2(input_array[:, 1], input_array[:, 0])
    output_array = np.transpose(np.vstack((alphas, input_array[:, 2])))
    return output_array

"""
def GPR_predicter(circle_dict, bin_dict, min_max_segment, max_depth):
    for circle in range(1, max_depth+1):
        seed_points_cart = circle_dict[circle]
        seed_points_cylinder = convert_coords(seed_points_cart)
        #Convert to (alpha, z) format
        for segment in range(min_max_segment[0], min_max_segment[1]+1):
            if segment == -2 and circle == 8:
                print("hej")
            bin_points_cart = bin_dict[(segment, circle)]
            bin_points_cylindrical = convert_coords(bin_points_cart)
            heights = GPR_predict(seed_points_cylinder, bin_points_cylindrical[:, 0], theta=0.1 * np.array([1, 1, 1])) #predicts on one bin at the time
            bin_points_cart[:, 2] = heights
            bin_dict[(segment, circle)] = bin_points_cart
    return
"""

#Takes in average bin height and points in bin:
#Outputs array of indexes for filtered points
def filter_points_hybrid(bin_average, heights, threshold):
    ground_indeces = np.argwhere(np.abs(bin_average - heights) < threshold)
    return ground_indeces.reshape(len(ground_indeces),)

def GPR_predicter(circle_dict, bin_dict, circle_mean_dict, threshold):
    bin_average_dict = {}
    filtered_points_in_bin_dict = {}
    for bin_tuple in bin_dict:
        segment = bin_tuple[0]
        circle = bin_tuple[1]
        mean = circle_mean_dict[circle]

        #Extract seed points for circle and points in the given bin
        seed_points_cart = circle_dict[circle]
        seed_points_cylinder = convert_coords(seed_points_cart)
        bin_points_cart = bin_dict[bin_tuple]
        bin_points_cylindrical = convert_coords(bin_points_cart)

        #Make zero mean for passing into GPR
        s = seed_points_cylinder[:, 1]
        seed_points_cylinder[:, 1] = seed_points_cylinder[:, 1] - mean
        bin_points_cylindrical[:, 1] = bin_points_cylindrical[:, 1] - mean

        #Prediction
        heights = GPR_predict(seed_points_cylinder, bin_points_cylindrical[:, 0])  # predicts on one bin at the time
        true_predicted_heights = heights + mean #Mean corrected heights of GPR-prediction
        average_height = np.mean(true_predicted_heights)
        bin_average_dict[bin_tuple] = average_height

        filtered_points_indexes = filter_points_hybrid(average_height, bin_points_cart[:, 2], threshold)
        filtered_heights = bin_points_cart[filtered_points_indexes, 2]

        filtered_points = np.hstack([bin_points_cart[filtered_points_indexes, :2], filtered_heights.reshape(len(filtered_heights), 1)])
        filtered_points_in_bin_dict[bin_tuple] = filtered_points

        bin_points_cart[:, 2] = true_predicted_heights
        bin_dict[bin_tuple] = bin_points_cart

    return bin_dict, bin_average_dict, filtered_points_in_bin_dict

def bin_dict2pcl(bin_dict):
    pcl_array = np.empty((0, 3), float)
    for bin in bin_dict:
        pcl_array = np.vstack([pcl_array, bin_dict[bin]])

    return pcl_array

# point_cloud, dx, dy, fract = 1/20, threshold = 0.1, delta = 0.2, iterations = 2, do_print = False
def hybrid_regression(point_cloud, d_alpha=(np.pi/2)/2, d_r=1, fract=1/20, threshold=0.3, iterations=5, large_slope=10):
    threshold = threshold
    d_alpha = d_alpha
    d_r = d_r
    bin_keys = get_bin_keys(points, d_alpha, d_r)
    bin_dict, min_bin_dict, seg_dict, min_seg_dict, circ_dict, min_circ_dict = make_dicts(bin_keys, points)
    max_depth_ind, min_max_segment = find_min_max_index(bin_keys)

    #RWLR-step
    SLm, circle_index_dictionary = make_seg_line(min_bin_dict, max_depth_ind, min_max_segment)
    sdm = filter_segments(SLm, fract=fract, iterations=iterations, large_slope=large_slope)

    #GPR-step
    SLn = make_circles(sdm, circle_index_dictionary, max_depth_ind, min_max_segment)
    circle_mean_dict = circle_mean(SLn)
    bin_dict_predictions, bin_average_dict, filtered_points_dict = GPR_predicter(SLn, bin_dict, circle_mean_dict,
                                                                                 threshold)
    # bin_dict_final = filter_points_hybrid(bin_average_dict, bin_dict, threshold)
    point_cloud_fit = bin_dict2pcl(bin_dict_predictions)
    ground_points = bin_dict2pcl(filtered_points_dict)


    #TODO: Add optimizer step
    return ground_points, bin_average_dict, point_cloud_fit

if False:
    # Generate point cloud
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
    points = pcl  # Used later - perhaps

    # Plot Input
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
    plt.show()

    # Hybrid Regression step
    gp, bad, pcf = hybrid_regression(pcl, d_r=0.1, d_alpha=0.02)
    # Note-crashes for spacing that is too fine (np.gradient - cannot seem to handle empty arrays)
    # Possible solution is to precheck the grid on the point_cloud - automatically compute d_r, d_alpha?

    # Plot Input
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
    # plt.show()

    # Plot Result Step
    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2], s=1)
    plt.show()

if True:
    # Generate point cloud
    n = 100
    x = np.linspace(-4.99, 4.99, n)
    y = np.linspace(-4.99, 4.99, n)
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
    points = pcl #Used later - perhaps

    #Plot Input
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
    plt.show()

    #Hybrid Regression step
    gp, bad, pcf = hybrid_regression(pcl, d_r=0.1, d_alpha=0.02, threshold=0.1)
    #Note-crashes for spacing that is too fine (np.gradient - cannot seem to handle empty arrays)
    #Possible solution is to precheck the grid on the point_cloud - automatically compute d_r, d_alpha?

    #Plot Input
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
    #plt.show()

    #Plot Result Step
    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2], s=1, c=[1,0,0])
    ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))
    plt.show()

"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(xarray, yarray, zarray, s=1, c=[0,0,0.5])
    ax.scatter(cgp[:, 0], cgp[:, 1], np.multiply(np.sin(cgp[:, 0]), np.cos(cgp[:, 1])), s=1, c=[0, 0, 0.5])
    # Wireframe requires data to be in matrix format
    ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))  # , s=1, c=[0, 0, 0.5])
    ax.scatter(cgp[:, 0], cgp[:, 1], cgp[:, 2], s=2, c=[1, 0, 0])
    plt.show()"""


"""
    points = np.array([[0.5, 0.5, 0],
                       [0, 0.5, 0.1],
                       [0.5, 0, 0.2],
                       [1.1, 0, 0.6],
                       [0.1, 0, 0],
                       [1, 1, 0.3],
                       [1, 2, 0.4],
                       [2, 1, 0.5],
                       [0.5, 1, 0.6],
                       [1, 0.5, 0.7],
                       [1, 0.6, 0.8],
                       [0.55, 1.1, 0.9]])


    #2D -scatterplot
    # plt.scatter(points[:,0], points[:,1])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    plt.show()

    """



"""
    #Assume the pointcloud has gone through initial pruning with r < 100 or similar
    # Max_depth
    threshold = 0.3
    d_alpha = (np.pi/2)/2
    d_r = 1
    bin_keys = get_bin_keys(points, d_alpha, d_r)
    bin_dict, min_bin_dict, seg_dict, min_seg_dict, circ_dict, min_circ_dict = make_dicts(bin_keys, points)
    max_depth_ind, min_max_segment = find_min_max_index(bin_keys)

    SLm, circle_index_dictionary = make_seg_line(min_bin_dict, max_depth_ind, min_max_segment)
    sdm = filter_segments(SLm, fract=1/20, iterations=2, large_slope=10)

    SLn = make_circles(sdm, circle_index_dictionary, max_depth_ind, min_max_segment)
    circle_mean_dict = circle_mean(SLn)

    bin_dict_predictions, bin_average_dict, filtered_points_dict = GPR_predicter(SLn, bin_dict, circle_mean_dict, threshold)
    #bin_dict_final = filter_points_hybrid(bin_average_dict, bin_dict, threshold)
    point_cloud_fit = bin_dict2pcl(bin_dict_predictions)
    ground_points = bin_dict2pcl(filtered_points_dict)
    #bin_indexes = np.unique(bin_keys, axis = 0)
    #print(bin_indexes)
    #print(bin_keys)

    #NEXT STEP IS:
    # *INCLUDE OPTIMIZER, MAKE OUTPUT INTO POINTCLOUD, THRESHOLD
    print("HEJHEJHEJ")
    """


#test grad_filter -seems to work
if False:
    r = np.array([0,1,2,3,4])
    z = np.array([0,0.1,1,1.2,1.3])
    val = grad_filter(z,r, large_slope=10)
    print(val)
