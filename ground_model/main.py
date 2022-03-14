from RWLR import *
import numpy as np
import matplotlib.pyplot as plt
from HybridRegression import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    #--------------- Ground Surface Point Filtering -------------------#
    # 2D-DEMO
    if False:
        # Generate point cloud
        n = 100
        x = np.linspace(-4.99, 4.99, n)
        y = np.linspace(-4.99, 4.99, n)
        xx, yy = np.meshgrid(x, y)
        degf = 3
        # The advantage of chisquare distr is that it strongly favours values on right hand side of peank than left - normalizing the peak to zero, the noise will have a strong bias to higher values
        # Thus it better represents both lidar noise and structures in pc
        noise = ((np.random.chisquare(df=degf, size=(n, n)) - (degf - 2)) / 5)# ** 2  # The method works a lot better with noise that is chisquare**2
        zz = np.multiply(np.sin(xx), np.cos(yy)) + noise


        def RMSE_fcn(xs, ys, zs):
            sq_err = (np.multiply(np.sin(xs), np.cos(ys)) - ys) ** 2
            return np.sqrt(sq_err.mean())


        # Add noise to height
        nrows, ncols = xx.shape
        xarray = np.reshape(xx, (nrows * ncols, 1))[:, 0]
        yarray = np.reshape(yy, (nrows * ncols, 1))[:, 0]
        zarray = np.reshape(zz, (nrows * ncols, 1))[:, 0]

        pcl = np.transpose(np.array([xarray, yarray, zarray]))

        if True:
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
        # fract=1/20, threshold=0.2, delta=0.05
        # Number of points seems most sensitive to delta - generally need larger delta for 2D ground case
        #Thinner stripes - better at bottom at top points
        cgp = GSPF(pcl, dx=0.2, dy=0.2, fract=1/20, threshold=0.1, delta=0.05, iterations=5, do_print=False)
        #INCREASING NUMBER OF ITERATIONS FOR THE LOWESS REGRESSIONS GAVE BETTER FITTING IN THIS CASE

        print(RMSE_fcn(cgp[:, 0], cgp[:, 1], cgp[:, 2]))
        # Plot results
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.scatter(xarray, yarray, zarray, s=1, c=[0,0,0.5])
        ax.scatter(cgp[:, 0], cgp[:, 1], np.multiply(np.sin(cgp[:, 0]), np.cos(cgp[:, 1])), s=1, c=[0, 0, 0.5])
        #Wireframe requires data to be in matrix format
        ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))#, s=1, c=[0, 0, 0.5])
        ax.scatter(cgp[:, 0], cgp[:, 1], cgp[:, 2], s=2, c=[1, 0, 0])
        plt.show()


    # 1D-DEMO
    if False:
        n = 2000
        stop = 50
        x = np.linspace(0, stop, num=n, endpoint=True)
        # x = np.append(x, stop)
        # print(np.shape(x))
        y = np.sin(x) + ((np.random.chisquare(df=3, size=n) - 1) / 5)  # **2#np.abs(np.random.normal(0,0.8, size = n))
        show_data(x,y,np.sin(x))
        y_fit, ground_inds = RWLR_iterator(y, x, fract=1 / 20, delta=0.05, threshold=0.2,
                                           display_result=False)  # decreasing fract makes it easier to find low points
        ground_x = x[ground_inds]
        ground_y = y[ground_inds]
        print(np.shape(y_fit.shape))
        show_data(ground_x, ground_y, np.sin(x)[ground_inds])


    # ---------------------- Hybrid Regression Technique ------------------------- #

    if True:
        # Generate point cloud

        #TEST TO SEE WHETHER HYBRID MODEL CAN HANDLE LARGE EMPTY SPACES
        n = 100
        x1 = np.linspace(0.01, 2, int(n/2))
        x2 = np.linspace(3, 4.99, int(n/2))
        x = np.hstack([x1, x2])

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


        """# Plot Input
        fig = plt.figure(1)
        fig.suptitle('Input Pointcloud')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
        plt.show()"""

        # Hybrid Regression step
        gp, ground_func, pcf = hybrid_regression(pcl, d_r=0.1, d_alpha=0.02, max_range=3)
        # Note-crashes for spacing that is too fine (np.gradient - cannot seem to handle empty arrays)
        # Possible solution is to precheck the grid on the point_cloud - automatically compute d_r, d_alpha?
        print(ground_func(np.array([[20, 20], [0.5, 0.5]])))

        fig = plt.figure(3)
        fig.suptitle('Hybrid Model filter points')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pcf[:, 0], pcf[:, 1], pcf[:, 2], s=1)
        #plt.show()



        # Plot Input
        fig = plt.figure(1)
        fig.suptitle('Input Pointcloud')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
        # plt.show()

        # Plot Result Step
        fig = plt.figure(2)
        fig.suptitle('Ground according to Hybrid Model')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2], s=1,  c=[1, 0, 0])
        ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))
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
        points = pcl  # Used later - perhaps

        # Plot Input
        fig = plt.figure(1)
        fig.suptitle('Input Pointcloud')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
        plt.show()

        # Hybrid Regression step
        gp, ground_func, pcf = hybrid_regression(pcl, d_r=0.1, d_alpha=0.02, threshold=0.1)
        # Note-crashes for spacing that is too fine (np.gradient - cannot seem to handle empty arrays)
        # Possible solution is to precheck the grid on the point_cloud - automatically compute d_r, d_alpha?


        # Plot Input
        fig = plt.figure(1)
        fig.suptitle('Input Pointcloud')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)
        # plt.show()

        # Plot Result Step
        fig = plt.figure(2)
        fig.suptitle('Ground according to Hybrid Model')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2], s=1, c=[1, 0, 0])
        ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))
        plt.show()


        #gradient = np.gradient(y, x) - y the f-values and x is the spacing
        #degrees = np.arctan(grad)*(180/np.pi)
        #Replace points with gradients >= 10deg with closest point