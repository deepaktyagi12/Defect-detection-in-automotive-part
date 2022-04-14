# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:47:18 2021

@author: dell
"""
import pickle
#from urllib import request
from itertools import chain
import collections
from libsvm import svmutil
import numpy as np
from scipy import signal
from scipy import special
from scipy import optimize
#import skimage
from sklearn.preprocessing import StandardScaler
import cv2

def normalize_kernel(kernel):
    """ This function is ued for kernel normalization.
    Parameters:
        kernel: Kernel size
    Returns:
        normalized value
    """
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n_var, sigma):
    y_var, x_var = np.indices((n_var, n_var)) - int(n_var/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x_var ** 2 + y_var ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')

def local_deviation(image, local_mean_var, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean_var ** 2 - sigma))

def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    c_var = 1/255 
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    # print(image.shape, kernel)
    mean_value = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, mean_value, kernel)
    return (image - mean_value) / (local_var + c_var)

def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
   
    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)

def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })

def asymmetric_generalized_gaussian(x, nu_var, sigma_l, sigma_r):
    def beta(sigma):
        return sigma * np.sqrt(special.gamma(1 / nu_var) / special.gamma(3 / nu_var))
   
    coefficient = nu_var / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu_var))
    f_var = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu_var)
       
    return np.where(x < 0, f_var(-x, sigma_l), f_var(x, sigma_r))


def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_rr_hat(r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(x, filter = lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        rr_hat = estimate_rr_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - rr_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(x, alpha, filter = lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))
   
    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
   
    alpha = estimate_alpha(x)
    sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
    sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)
   
    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r)
   
    return alpha, mean, sigma_l, sigma_r

def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
    """ This function calculates the brisque features.
    Parameters:
        image: Any input image
        kernel_size: Kernel is required for Gaussian distribution
        sigma: Variance
    Returns:
        An array of flatten brisque features
    """
    def calculate_features(coefficients_name, coefficients):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]
       
        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
   
    mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)
   
    features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
    flatten_features = list(chain.from_iterable(features))
    return np.array(flatten_features, dtype=object)
'''
def load_image(url):
    """ This function loads image
    Parameters:
        url: The path of the input image
    Returns:
        Reads the image from the given path
    """
    image_stream = request.urlopen(url)
    return skimage.io.imread(image_stream, plugin='pil')
'''


"""
def plot_histogram(x, label):
    n, bins = np.histogram(x.ravel(), bins=50)
    n = n / np.max(n)
    #plt.plot(bins[:-1], n, label=label, marker='o')
"""    
def scale_features(features):
    with open('normalize.pickle', 'rb') as handle:
        scale_params = pickle.load(handle)
   
    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])
   
    return -1 + (2.0 / (max_ - min_) * (features - min_))

# define standard scaler
def my_scale(features):
    """ This function performs the features scaling.
    Parameters:
        features: Input features acquired from MSCN coeff
    Returns:
        Scaled features
    """
    with open('normalize.pickle', 'rb') as handle:
        scale_params = pickle.load(handle)

    scaler = StandardScaler()
    # transform data
    return scaler.fit_transform(scale_params)

def calculate_image_quality_score(brisque_features):
    """ This function applies LIBSVM for prediction, and uses the model saved in
    'brisque_svm.txt' file.
    Parameters:
        brisque_features: Features (scaled) extracted from MSCN coefficients
    Returns:
        model: Model from SVM file
        x: Labels
        prob_estimates: Probability estimates
    """
    model = svmutil.svm_load_model('brisque_svm.txt')
    scaled_brisque_features = brisque_features.reshape(-1, 1)

    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
   
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
   
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)

def final_quality_score(image_path):
    """ 
    This function yields the final image quality index.
    Parameters:  
        new_image: The path of the input image
    Returns: quality_score_original The image quality scoreb
    """
    #mscn_coefficients = calculate_mscn_coefficients(new_image, 7, 7/6)
    #coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    new_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    brisque_features = calculate_brisque_features(new_image, kernel_size=7, sigma=7/6)

    downscaled_image = cv2.resize(new_image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)

    brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
    quality_score_original = calculate_image_quality_score(brisque_features)
    quality_score_original = 100-quality_score_original
    
    return quality_score_original