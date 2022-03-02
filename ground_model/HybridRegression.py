import numpy as np


def make_cov_fcn(x1, x2, l_const=1, sig_f=1, sig_n=1, is_noise=False):
    xx1, xx2 = np.meshgrid(x2, x1) #rows correspond to x1 and columns to x2
    l_const = 2*(l_const**2)
    K = (sig_f**2) * np.exp((-(xx1 - xx2)**2)/l_const)
    if is_noise and len(x1) == len(x2):
        K = K + np.identity(len(x1))*sig_n**2
    return K

#Theta is presumed to be on form: (l_const, sig_f, sig_n)
#seed is ndarray with (alpha, z) elements
#bin_points array of alpha:s
def GPR_predict(seed, bin_points, theta = 0.1*np.array([1, 1, 1])):
    l_const = theta[0]
    sig_f = theta[1]
    sig_n = theta[2]
    z_seed = np.transpose(seed[:, 1])
    cov_1 = make_cov_fcn(bin_points, seed[:, 0], l_const=l_const, sig_f=sig_f, sig_n=sig_n)
    cov_2 = make_cov_fcn(seed[:, 0], seed[:, 0], l_const=l_const, sig_f=sig_f, sig_n=sig_n, is_noise=True)
    z_predict = (cov_1.dot(cov_2)).dot(z_seed)
    return z_predict

a = np.array([1])
b = np.array([1, 2, 3])
#print(make_cov_fcn(a, b))

#print(make_cov_fcn(b,b,is_noise=True))

s = np.array([[0.1, 4],
             [0.2, 4.3],
             [0.3, 5]])
bin_points = np.array([0.22])
print(GPR_predict(s, bin_points))