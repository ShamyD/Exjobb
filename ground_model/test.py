import numpy as np
#import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
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




plt.scatter(points[:,0], points[:,1])
#plt.show()

#Converts np-array of points into np-array of their bins
def get_bin_keys(points, d_alpha, d_r):
    segs = np.ceil(np.arctan2(points[:, 1],points[:, 0])/d_alpha)
    circs = np.ceil(np.linalg.norm(points[:, :2], axis = 1)/d_r)
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
        min_seg_dict[key[0]] = points_in_seg[0,:]
    return bin_dict, min_bin_dict, seg_dict, min_seg_dict

def get_min_PL(bin_dict):
    return

d_alpha = (math.pi/2)/2
d_r = 1
bin_keys = get_bin_keys(points, d_alpha, d_r)
bin_dict, min_bin_dict, seg_dict, min_seg_dict = make_dicts(bin_keys, points)

#TODO Most likely points are selected as initial seeds

#TODO Make sure there are no gaps with no values - extrapolate from neighbours


#bin_indexes = np.unique(bin_keys, axis = 0)
#print(bin_indexes)
#print(bin_keys)

