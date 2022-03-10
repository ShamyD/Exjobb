from RWLR import *
import numpy as np
import matplotlib.pyplot as plt
from HybridRegression import *

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

#HYBRID MODEL:
hybrid_ground_points, bad, pcf = hybrid_regression(pcl, d_r=0.1, d_alpha=0.02, threshold=0.05)

#GROUND SURFACE POINTS FILTERING_ONLY RWLR:
GSPF_ground_points = GSPF(pcl, dx=0.2, dy=0.2, fract=1/20, threshold=0.1, delta=0.05, iterations=5, do_print=False)

# Plot Input
fig = plt.figure(1)
fig.suptitle('Input Pointcloud')
ax = fig.add_subplot(projection='3d')
ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=1)

#Plot Hybrid model's ground points
fig = plt.figure(2)
fig.suptitle('Ground according to Hybrid Model')
ax = fig.add_subplot(projection='3d')
ax.scatter(hybrid_ground_points[:, 0], hybrid_ground_points[:, 1], hybrid_ground_points[:, 2], s=1, c=[1, 0, 0])
ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))

#Plot GSPF model's ground points
fig = plt.figure(3)
fig.suptitle('Ground according to GSPF')
ax = fig.add_subplot(projection='3d')
ax.scatter(GSPF_ground_points[:, 0], GSPF_ground_points[:, 1], GSPF_ground_points[:, 2], s=1, c=[1, 0, 0])
ax.plot_wireframe(xx, yy, np.multiply(np.sin(xx), np.cos(yy)))

plt.show()