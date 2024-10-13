import os
import numpy as np
from scipy.integrate import quad

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def l2_H1_err_l2_H1_relative_G(coeff_1, coeff_2, start=0, end=1):
    # Here we use sine function as basis
    def function_error(x):
        weight_list_1 = (np.arange(coeff_1.shape[1]) + 1) * np.pi
        sin_list_1 = np.sin(weight_list_1*x)
        
        weight_list_2 = (np.arange(coeff_2.shape[1]) + 1) * np.pi
        sin_list_2 = np.sin(weight_list_2*x)
        return (np.sum(coeff_1 * sin_list_1) - np.sum(coeff_2 * sin_list_2) ) ** 2
        
    def gradient_error(x):
        weight_list_1 = (np.arange(coeff_1.shape[1]) + 1) * np.pi
        cos_list_1 = np.cos(weight_list_1*x)
        
        weight_list_2 = (np.arange(coeff_2.shape[1]) + 1) * np.pi
        cos_list_2 = np.cos(weight_list_2*x)
        return (np.sum(coeff_1 * cos_list_1 * weight_list_1) - np.sum(coeff_2 * cos_list_2 * weight_list_2)) **2
    
    def function_2(x):
        weight_list_2 = (np.arange(coeff_2.shape[1]) + 1) * np.pi
        sin_list_2 = np.sin(weight_list_2*x)
        return (np.sum(coeff_2 * sin_list_2)) ** 2
    
    def gradient_2(x):
        weight_list_2 = (np.arange(coeff_2.shape[1]) + 1) * np.pi
        cos_list_2 = np.cos(weight_list_2*x)
        return (np.sum(coeff_2 * cos_list_2 * weight_list_2)) **2
    
    
    inegral_func, _ = quad(function_error, 0, 1)
    inegral_grad, _ = quad(gradient_error, 0, 1)
    err_l2 = inegral_func ** (1/2)
    err_h1 =  (inegral_func + inegral_grad) ** (1/2)

    inegral_func_2, _ = quad(function_2, 0, 1)
    inegral_grad_2, _ = quad(gradient_2, 0, 1)  
    value_2_l2 = inegral_func_2 ** (1/2)
    value_2_h2 = (inegral_func_2 + inegral_grad_2) ** (1/2)
    
    return err_l2, err_l2 / value_2_l2, err_h1, err_h1 / value_2_h2
    