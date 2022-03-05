import numpy as np
from scipy.optimize import minimize

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


# --------- Optimizer functions -----------------------#
def make_partial_grad_matrices(x, theta):
    l_const = theta[0]
    sig_f = theta[1]
    sig_n = theta[2]
    xx1, xx2 = np.meshgrid(x, x)
    d_Kl = (sig_f ** 2) * np.exp((-(xx1 - xx2) ** 2) / (2*l_const**2))*((xx1 - xx2)**2)/(l_const**3)
    d_Ksig_f = 2*sig_f*np.exp((-(xx1 - xx2) ** 2) / (2*l_const**2))
    d_Ksig_n = 2*sig_n*np.identity(len(x))
    return d_Kl, d_Ksig_f, d_Ksig_n

#data_dict[n] = ndarray of data points (alpha, z) for n:th circle
#Total of N circles [0-N-1]
def hyperparameter_optimizer(data_dict, N, initial_guess):

    #Nested function so that the function to be minimized only takes theta as input
    def log_likelihood_loss(theta):
        loss_vect = np.zeros([len(data_dict), ])
        for group in range(len(data_dict)): #For every circle(group), calculate loss and add together
            data = data_dict[group]
            z = data[:, 1]
            alpha = data[:, 0]
            cov_matrix = make_cov_fcn(alpha, alpha, theta[0], theta[1], theta[2], is_noise=True)
            loss_vect[group] = 0.5 * ((z.dot(cov_matrix)).dot(z) + np.log(np.linalg.det(cov_matrix)) + np.log(2*np.pi)*len(z))


        return np.sum(loss_vect)

    def loss_gradient(theta):
        grad_vect = np.zeros([len(data_dict), 3])
        for group in range(len(data_dict)): #I've chosen the more memory intensive method since it is clearer and likely less computationally heavy
            data = data_dict[group]
            z = data[:, 1]
            alpha = data[:, 0]
            cov_matrix = make_cov_fcn(alpha, alpha, theta[0], theta[1], theta[2], is_noise=True)
            cov_inv = np.linalg.inv(cov_matrix)
            a_matrix = cov_inv.dot(z)
            common_matrix = a_matrix.dot(a_matrix) - cov_inv

            Kl, Ksigf, Ksign = make_partial_grad_matrices(alpha, theta)
            
            #Compute hyperparameter-specific components
            grad_l = 0.5*np.trace(common_matrix.dot(Kl))
            grad_sigf = 0.5*np.trace(common_matrix.dot(Ksigf))
            grad_sign = 0.5*np.trace(common_matrix.dot(Ksign))
            grad_vect[group, :] = np.array([-grad_l, -grad_sigf, -grad_sign])

        return np.sum(grad_vect, axis=0)

    result = minimize(log_likelihood_loss, initial_guess, jac=loss_gradient, tol=0.001)
    print(result)
    #ADD EXCEPTION HANDLING IF BAD INPUT IS GIVEN - RESULT CONTAINS GOOD INFORMATION TO USE
    return result.x




#------------------ Computations to be added in main ------------------------------------#
if True:
    a = np.array([1])
    b = np.array([1, 2, 3])
    #print(make_cov_fcn(a, b))
    #print(make_cov_fcn(b,b,is_noise=True))
    s = np.array([[0.1, 1],
                  [0.2, 0.4],
                  [0.3, -0.3],
                  [0.4, -1]])
    bin_points = np.array([0.22])
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

if True:
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

    init = np.array([0.1, 0.1, 0.1])
    print(hyperparameter_optimizer(data, 2, init))