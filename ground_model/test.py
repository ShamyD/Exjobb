import numpy as np
#import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

points = np.array([[0.5, 0.5, 0],
                   [0, 0.5, 0.1],
                   [0.5, 0, 0.2],
                   [1, 1, 0.3],
                   [1, 2, 0.4],
                   [2, 1, 0.5],
                   [0.5, 1, 0.6],
                   [1, 0.5, 0.7],
                   [1, 0.6, 0.8],
                   [0.55, 1.1, 0.9]])

#print(np.arctan2(points[:,1],points[:,0])/(math.pi/6))
#print(np.ceil(np.arctan2(points[:,1],points[:,0])/(math.pi/6)))





#plt.scatter(points[:,0], points[:,1])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:, 2], s=1)
plt.show()

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
        min_bin_dict[tuple(key)] = points_in_bin[0,:]

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


def find_min_max_segment_index():

    return
#Takes in min_bin_dict and for every bin in a segment, retreives the smallest values (min_bin_dict)
#Prodces a dictionary of arrays for every segment
# If a bin between 0 and max_depth_index has no value, this is skipped. all circles - [0:max_depth_index]
def make_seg_line(min_bin_dict, max_depth_index, min_max_segment):
    SLm_dict = {} #SLm_dict[m] gives array of points for the m:th segment
    for segment in range(min_max_segment[0], min_max_segment[1]+1):
        segment_array = np.empty((0, 3), float)
        for circle in range(max_depth_index+1):
            if (segment, circle) in min_bin_dict:
                segment_array = np.vstack([segment_array, min_bin_dict[segment, circle]])
        SLm_dict[segment] = segment_array

    return SLm_dict
    #for bin in range()

#Take in a dictionary of segment ndarrays, filters these with the RLWR-regression and smoothes the result with a gradient filter
def filter_segments(SLm_dict, fract):
    lowess = sm.nonparametric.lowess
    for segment in SLm_dict:
        segment_points = SLm_dict[segment]
        z = segment_points[:, 2]
        r = np.linalg.norm(segment_points[:, :2], axis=1)
        regr = lowess(z, r, frac=fract, it=iterations, return_sorted=False)
        #Gradient filter on points
        gradients = np.gradient(regr, r) #- regr the f-values and r is the spacing
        degrees = np.arctan(gradients)*(180/np.pi)

        large_slope_indexes = SDKLÖJHGFDSDGHKLÖLKJHGFDDFGHJKLÖLKJHGFDDFGHJKLÖLKJHGFDFGHJKKJHFDFGHJKLLKJHGFDFGHJKLLKJHGFGHJKLÖLKJHGGFDFGHJKJHGFFGHJKLKJHGFDFGHJKJVCVHJHTRTJ
        #degrees = np.arctan(grad)*(180/np.pi)
        #Replace points with gradients >= 10deg with closest point
    return

def get_min_PL(bin_dict):
    return

#Assume the pointcloud has gone through initial pruning with r < 100 or similar
# Max_depth
d_alpha = (math.pi/2)/2
d_r = 1
bin_keys = get_bin_keys(points, d_alpha, d_r)
bin_dict, min_bin_dict, seg_dict, min_seg_dict, circ_dict, min_circ_dict = make_dicts(bin_keys, points)
max_min_segment = [0, 2]
max_depth_ind = 3
#TODO: find good ways of finding max and min segment and max depth
SLm = make_seg_line(min_bin_dict, max_depth_ind, max_min_segment)



print(min_seg_dict)
print("-------------")
print(min_bin_dict)


#bin_indexes = np.unique(bin_keys, axis = 0)
#print(bin_indexes)
#print(bin_keys)

