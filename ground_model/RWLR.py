import numpy as np
import scipy as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import copy
from matplotlib import cm

def show_data(inp,out, fit = None):
    plt.scatter(inp,out,s = 4, c=[1,0,0])
    if fit is not None:
        plt.scatter(inp,fit,s = 0.5, c=[0,0,1])
    plt.show()

#Bisquare Weight function
def BRW(r):
    MAD = np.median(np.abs(r)) #median absolute distance
    r_star = r/(6*MAD)
    w = np.zeros(np.shape(r_star))
    less_than_zero = np.abs(r_star) < 1
    w[less_than_zero] = (1 - r_star[less_than_zero]**2)**2
    return w

def RMSE(predictions, targets):
    return np.sqrt(((predictions-targets)**2).mean())

#Requires z_values being one dimensional
def low_outlier_removal(z_value, x_values, frac):
    k = int(np.floor(frac*np.size(z_value))) #make sure k is an integer
    x_temp = x_values.reshape((np.size(x_values), 1)) #KNN-needs this format np.array([[.], [.],.....[.]])
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(x_temp)
    distances, indices = nbrs.kneighbors(x_temp)

    z_replace = copy.copy(z_value) #For only adding changes we are interested of

    for i, value in enumerate(z_value):
        #Compare every value with min value for neighbours - if itself is min replace value with second min in neighbourhood
        indexes = indices[i, :]
        lowest_val_index = np.argmin(z_value[indexes]) #index of indexes-array
        lowest_index = indexes[lowest_val_index]
        if lowest_index == i:
            """
            print(i)
            #Should give the smallest value of neighbours that is not itself
            print(indexes)
            print(np.delete(indexes, np.where(indexes == i)))
            print(z_value[np.delete(indexes, np.where(indexes == i))])
            print(np.argmin(z_value[np.delete(indexes, np.where(indexes == i))]))
            print(z_value[np.argmin(z_value[np.delete(indexes, np.where(indexes == i))])])

            print("------------------------")
            print(np.amin(z_value[np.delete(indexes, np.where(indexes == i))]))
            """
            z_replace[i] = np.amin(z_value[np.delete(indexes, np.where(indexes == i))])
            #z_value[np.argmin(z_value[np.delete(indexes, np.where(indexes == i))])]
        #print(z_replace)
    return z_replace

def filter_points(fitted_line, true_y, threshold):
    ground_indeces = np.where(np.abs(fitted_line - true_y) < threshold)
    return ground_indeces

#Performs RWLR-iterations according to article for a single line/stripe
def RWLR_iterator(dep_var, indep_var, fract, delta, threshold, iterations = 2,display_progress = False, display_result = False, do_print = False):
    #Y is dependent variable and X is independent
    z = dep_var
    x = indep_var
    lowess = sm.nonparametric.lowess
    # Rescale data
    min_height = np.amin(z)
    z = z - min_height

    old_rmse = 0
    # Iteratively reestimate curve and push-down remove points that are too high
    while True:
        # First column are the sorted x-values
        #show_data(x,z)
        regr = lowess(z, x, frac=fract, it=iterations, return_sorted=False)#[:,1] #Is this a full RLWR? - second column is fitted values and first is sorted x-values
        #frac is the fraction of data used for estimation at every point: k = frac*n


        #TASK 1: calc residuals
        resids = z - regr

        #TASK 2: Classification of points in over and under fitted line
        points_above_line = resids > 0

        #TASK 3: Weight points above fitted line with Bisquare robust weight function and give the other points a weight of one
        w = np.ones(np.shape(resids))
        #The function returns zero for absolute residuals larger than 1
        w[points_above_line] = BRW(resids[points_above_line])


        z = np.multiply(z, w)
        if display_progress:
            show_data(x, z, regr)
        #print(type(y))
        z = low_outlier_removal(z, x, fract)

        rmse = RMSE(regr, z)
        if do_print:
            print("RMSE: " + str(rmse))
        if np.abs(rmse - old_rmse) < delta:
            if do_print:
                print("Delta: " + str(np.abs(old_rmse - rmse)))
                print("Breaking")
            break
        else:
            if do_print:
                print("Delta: " + str(np.abs(old_rmse - rmse)))
            old_rmse = rmse


    #Rescale
    regr = regr + min_height
    ground_indeces = filter_points(regr, dep_var, threshold) #Returns the points considered ground points
    #plt.scatter(x, w)
    #plt.show()
    if display_result:
        z = z + min_height
        show_data(x, z, regr)
    return regr, ground_indeces

#The full ground extraction algorithm in the RLWR article. Takes
def RWLR_combiner(pos_dict, val_dict, compl_axis_dict, N, M, fract, threshold, delta, iterations, do_print = False):
    #pos_dict line to x_values or y_values: pos_dict[0:N-1] for x and pos_dict[N: N+M-1] for y
    #val_dict line to z_values: same indexing

    ground_points_x = np.empty((0, 3), float)
    ground_points_y = np.empty((0, 3), float)
    for dict_key in range(N+M):
        if do_print:
            print("iteration " + str(dict_key+1) + " of " + str(M+N))
        pos_array = pos_dict[dict_key]
        val_array = val_dict[dict_key]
        compl_axis = compl_axis_dict[dict_key]
        _, ground_inds = RWLR_iterator(val_array, pos_array, fract, threshold, delta, iterations) #Return ground indexes for the given stripe
        if dict_key < N:
            #Create matrix with [x,y,z] - rows
            ground_points = np.transpose([pos_array[ground_inds], compl_axis[ground_inds], val_array[ground_inds]])
            ground_points_x = np.vstack([ground_points_x, ground_points])
        else:
            #Create matrix with [x,y,z] - rows
            a = compl_axis[ground_inds]
            ground_points = np.transpose([compl_axis[ground_inds], pos_array[ground_inds], val_array[ground_inds]])
            ground_points_y = np.vstack([ground_points_y, ground_points])

    # Return Intersection of the points - enables check row-wise for intersect1d and then reshapes data in ndarray
    nrows, ncols = ground_points_x.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [ground_points_x.dtype]}
    common_ground_points = np.intersect1d(ground_points_x.view(dtype), ground_points_y.view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    common_ground_points = common_ground_points.view(ground_points_x.dtype).reshape(-1, ncols)

    return common_ground_points

#Divide data into stripes
def get_stripe_keys(points, d_x, d_y):
    x_stripe = np.ceil(points[:, 1]/d_y) #Stripes in which the x-direction will be studied - Horizontal
    y_stripe = np.ceil(points[:, 0]/d_x)
    stripe_keys = np.transpose(np.array([x_stripe.astype(int), y_stripe.astype(int)]))
    return stripe_keys

#Create dictionaries with data divided into stripes
def make_dicts(stripe_keys, points):
    #Number of stripes in each direction
    xmin = np.amin(stripe_keys[:, 0])
    xmax = np.amax(stripe_keys[:, 0])
    ymin = np.amin(stripe_keys[:, 1])
    ymax = np.amax(stripe_keys[:, 1])
    N = int(xmax - xmin + 1)
    M = int(ymax - ymin + 1)
    #stripe_indexes = np.unique(stripe_keys, axis=0)

    pos_dict = {}
    val_dict = {}
    compl_axis_dict = {}
    for stripe_index in range(xmin, xmax + 1):
        dict_index = stripe_index - xmin #To place in correct index of dictionaries
        indexes = np.where(stripe_keys[:, 0] == stripe_index)[0]
        points_in_stripe = points[indexes, :]
        pos_dict[dict_index] = points_in_stripe[:, 0] #Add x-axis as pos index
        compl_axis_dict[dict_index] = points_in_stripe[:, 1] #y-axis is complementary axis
        val_dict[dict_index] = points_in_stripe[:, 2]
    for stripe_index in range(ymin, ymax + 1):
        dict_index = stripe_index - ymin + N
        indexes = np.where(stripe_keys[:, 1] == stripe_index)[0]
        points_in_stripe = points[indexes, :]
        pos_dict[dict_index] = points_in_stripe[:, 1] #Add y-axis as pos index
        compl_axis_dict[dict_index] = points_in_stripe[:, 0] #y-axis is complementary axis
        val_dict[dict_index] = points_in_stripe[:, 2]

    return pos_dict, compl_axis_dict, val_dict, N, M

#Ground Surface Points Filtering - Main function from paper
def GSPF(point_cloud, dx, dy, fract = 1/20, threshold = 0.1, delta = 0.2, iterations = 2, do_print = False):
    stripe_keys = get_stripe_keys(point_cloud, 0.5, 0.5)
    position_dict, complementary_position_dict, value_dict, N, M = make_dicts(stripe_keys, point_cloud)
    #fract=1/20, threshold=0.2, delta=0.05
    #Number of points seems most sensitive to delta - generally need larger delta for 2D ground case
    common_ground_points = RWLR_combiner(position_dict, value_dict, complementary_position_dict, N, M, fract=fract, threshold=threshold, delta=delta, iterations = iterations, do_print=do_print)
    return common_ground_points

# ----------------------------------- Commands start here --------------------------------------------------------------#
"""
# 2D-demo
if True:
    #Generate point cloud
    n = 100
    x = np.linspace(-4.99, 4.99, n)
    y = np.linspace(-4.99, 4.99, n)
    xx, yy = np.meshgrid(x, y)
    degf = 3
    #The advantage of chisquare distr is that it strongly favours values on right hand side of peank than left - normalizing the peak to zero, the noise will have a strong bias to higher values
    #Thus it better represents both lidar noise and structures in pc
    noise = ((np.random.chisquare(df = degf, size = (n,n))-(degf-2))/5)**2 #The method works a lot better with noise that is chisquare**2
    zz = np.multiply(np.sin(xx),np.cos(yy)) + noise

    def RMSE_fcn(xs, ys, zs):
        sq_err = (np.multiply(np.sin(xs), np.cos(ys)) - ys)**2
        return np.sqrt(sq_err.mean())

    #Add noise to height
    nrows, ncols = xx.shape
    xarray = np.reshape(xx, (nrows*ncols, 1))[:, 0]
    yarray = np.reshape(yy, (nrows*ncols, 1))[:, 0]
    zarray = np.reshape(zz, (nrows*ncols, 1))[:, 0]

    pcl = np.transpose(np.array([xarray, yarray, zarray]))

    if False:
        ####### PLOTTING ################
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xarray, yarray, zarray, s=1)
        plt.show()

    #pcl,1,1
    #stripe_keys = get_stripe_keys(pcl, 0.5, 0.5)
    #pd, cd, vd, N, M = make_dicts(stripe_keys, pcl)
    #fract=1/20, threshold=0.2, delta=0.05
    #Number of points seems most sensitive to delta - generally need larger delta for 2D ground case
    #cgp = RWLR_combiner(pd, vd, cd, N, M, fract=1/20, threshold=0.1, delta=0.2, do_print=True)

    cgp = GSPF(pcl, dx=0.5, dy=0.5, fract=1/20, threshold=0.1, delta=0.2, do_print=False)

    print(RMSE_fcn(cgp[:, 0], cgp[:, 1], cgp[:, 2]))
    #Plot results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(xarray, yarray, zarray, s=1, c=[0,0,0.5])
    ax.scatter(cgp[:, 0], cgp[:, 1], np.multiply(np.sin(cgp[:, 0]),np.cos(cgp[:, 1])), s=1, c=[0,0,0.5])
    ax.scatter(cgp[:, 0], cgp[:, 1], cgp[:, 2], s=2, c=[1,0,0])
    plt.show()

    For debugging
    pcl = np.array([[0.5, 0.2, 0.6],
                    [0.4, 0.3, 0.7],
                    [-0.2, 0.5, 0.7],
                   [-0.3, -0.2, -1],
                    [0.2, -0.9, 2]])
    stripe_keys = get_stripe_keys(pcl, 1, 1)
    pd, cd, vd, N, M = make_dicts(stripe_keys, pcl)



# 1D-demo
if False:
    n = 2000
    stop = 50
    x = np.linspace(0, stop, num = n, endpoint=True)
    #x = np.append(x, stop)
    #print(np.shape(x))
    y = np.sin(x) + ((np.random.chisquare(df = 3, size = n)-1)/5)#**2#np.abs(np.random.normal(0,0.8, size = n))
    #show_data(x,y,np.sin(x))
    y_fit, ground_inds = RWLR_iterator(y,x,fract = 1/20, delta = 0.05, threshold = 0.2, display_result=False) #decreasing fract makes it easier to find low points
    ground_x = x[ground_inds]
    ground_y = y[ground_inds]
    print(np.shape(y_fit.shape))
    show_data(ground_x, ground_y, np.sin(x)[ground_inds])

"""