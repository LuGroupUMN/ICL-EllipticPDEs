import numpy as np
from scipy.sparse import diags
from scipy.special import p_roots
import matplotlib.pyplot as plt

# generate data:
# generate data:
def generate_colors(n, colormap='jet'): #'viridis'
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / n) for i in range(n)]
        return colors
    
def KL_rf(x, tau=5, alpha=3, dim=1, order=50):
    bx =  0
    for i in range(order):
        lambda_i = ((i+1) ** 2) * (np.pi ** 2) + tau
        phi_i = np.sin((i+1) * np.pi * x)
        xi = np.random.normal(0, 1, x.shape[0])
        phi_i = xi * phi_i
        bx += (lambda_i ** (-alpha/2)) * phi_i
    return np.exp(bx)

def KL_rf_tensor(x, tau=5, alpha=3, dim=1, order=50):
    bx =  0
    for i in range(order):
        lambda_i = ((i+1) ** 2) * (np.pi ** 2) + tau
        phi_i = np.sin((i+1) * np.pi * x)
        xi = np.random.normal(0, 1, size=(x.shape[0],dim))
        phi_i = np.einsum("a,ab->ab", phi_i, xi) # xi * phi_i 
        bx += (lambda_i ** (-alpha/2)) * phi_i
    return np.exp(bx)

def KL_rf_tensor_ne(x, tau=5, alpha=3, scaler=1, dim=1, order=50):
    bx =  0
    for i in range(order):
        lambda_i = ((i+1) ** 2) * (np.pi ** 2) + tau
        phi_i = np.sin((i+1) * np.pi * x)
        xi = np.random.normal(0, 1, size=(x.shape[0],dim))
        phi_i = np.einsum("a,ab->ab", phi_i, xi) # xi * phi_i 
        bx += (lambda_i ** (-alpha/2)) * phi_i * scaler
    return bx

def gauss_root_weight(n,a,b):
    [x,w] = p_roots(n+1)
    x_scaled = 0.5*(b-a)*x+0.5*(b+a)
    #G=0.5*(b-a)*sum(w*f(0.5*(b-a)*x+0.5*(b+a)))
    return x_scaled, w

def get_a_and_V(x_np, tau=5, alpha=3,left=1,right=2):
    # default of distribution of V is [1,2]
    v_value = np.random.uniform(low=left, high=right, size=x_np.shape[0])
    return KL_rf(x_np, tau=tau, alpha=alpha), v_value

def get_A(x_np, a_np, v_np, input_dim):
    #A_result = np.zeros(input_dim)
    i_pi_cos_input_1 = np.array([np.sqrt((1-0) / 2) * np.sqrt(2) * i * np.pi * np.cos(i * np.pi * x_np) * a_np for i in range(1,input_dim+1)])
    i_pi_cos_input_2 = np.array([np.sqrt((1-0) / 2) * np.sqrt(2) * i * np.pi * np.cos(i * np.pi * x_np) for i in range(1,input_dim+1)])
    sin_input_1 = np.array([np.sqrt((1-0) / 2) * np.sqrt(2) * np.sin(i * np.pi * x_np) * v_np for i in range(1,input_dim+1)])
    sin_input_2 = np.array([np.sqrt((1-0) / 2) * np.sqrt(2) * np.sin(i * np.pi * x_np)  for i in range(1,input_dim+1)])

    A_result = np.einsum("ab,bc->ac", i_pi_cos_input_1, i_pi_cos_input_2.T) + np.einsum("ab,bc->ac", sin_input_1, sin_input_2.T)
    return A_result 
    
def get_random_invertible_matrix(input_dim, total_num, matrix_type="laplacian",tau=5,alpha=3,left=1,right=2,order=0,seed_value=100):
    np.random.seed(seed_value)
    if matrix_type=="diag":
        A_list = []
        for i in range(total_num):
            diagonal_entries = np.random.uniform(low=1.0, high=2.0, size=input_dim)
            A_temp = np.diag(diagonal_entries)
            A_list.append(A_temp)
        return A_list
    elif matrix_type == "Galerkin":
        A_list = []
    
        x_np, w_np = gauss_root_weight(2*input_dim+1,0,1)
        
        for i in range(total_num):
            a_np, v_np = get_a_and_V(x_np, tau=tau, alpha=alpha,left=left,right=right)
            A_temp = get_A(x_np, a_np * w_np, v_np * w_np, input_dim)
            A_list.append(A_temp)
        return A_list
    elif matrix_type == "Galerkin_constanta":
        A_list = []
        
        if order == 0:
            x_np, w_np = gauss_root_weight(2*input_dim+1,0,1)
        else:
            x_np, w_np = gauss_root_weight(order,0,1)
        
        for i in range(total_num):
            v_np = np.random.uniform(low=left, high=right, size=x_np.shape[0])
            A_temp = get_A(x_np, w_np, v_np * w_np, input_dim)
            A_list.append(A_temp)
        return A_list
    
    elif matrix_type == "Galerkin_constanta_cV":
        A_list = []
        
        if order == 0:
            x_np, w_np = gauss_root_weight(2*input_dim+1,0,1)
        else:
            x_np, w_np = gauss_root_weight(order,0,1)
        
        for i in range(total_num):
            #v_np = np.random.uniform(low=1.0, high=2.0, size=x_np.shape[0])
            v_np = np.random.uniform(low=1.0, high=2.0) * np.array([1] * x_np.shape[0])
            A_temp = get_A(x_np, w_np, v_np * w_np, input_dim)
            A_list.append(A_temp)
        return A_list
    
    elif matrix_type == "symetric":
        # (A + A.T) / 2
        return 0
    elif matrix_type == "laplacian":
        A_list = []
        k = [-np.ones(input_dim-1),2*np.ones(input_dim),-np.ones(input_dim-1)]
        offset = [-1,0,1]
        Lap_basic = diags(k,offset).toarray()
        for i in range(total_num):
            #diagonal_entries = np.random.uniform(low=0.0, high=1.0, size=input_dim)
            diagonal_entries = np.random.uniform(low=1.0, high=2.0, size=input_dim)
            A_temp = np.diag(diagonal_entries)
            A_list.append((A_temp+Lap_basic * (input_dim**2)))
        return A_list
    elif matrix_type == "laplacian_c":
        A_list = []
        k = [-np.ones(input_dim-1),2*np.ones(input_dim),-np.ones(input_dim-1)]
        offset = [-1,0,1]
        Lap_basic = diags(k,offset).toarray()
        for i in range(total_num):
            #diagonal_entries = np.random.uniform(low=0.0, high=1.0, size=input_dim)
            diagonal_entries = np.random.uniform(low=1.0, high=2.0) * np.array([1] * input_dim)
            A_temp = np.diag(diagonal_entries)
            A_list.append((A_temp+Lap_basic * (input_dim**2)))
        return A_list
    elif matrix_type == "laplacian_rf":
        A_list = []
        grid_np = np.arange(input_dim) / input_dim
        period_bound = [i for i in range(1,input_dim)] + [0]
        
        offset = [-1,0,1]
        for i in range(total_num):
            diag_item = KL_rf(grid_np, tau=tau, alpha=alpha, dim=1, order=20)
            k = [-diag_item[:-1],diag_item*2, -diag_item[1:]]

            Lap_rf = diags(k,offset).toarray()
            diagonal_entries = np.random.uniform(low=1.0, high=2.0, size=input_dim)
            A_temp = np.diag(diagonal_entries)
            A_list.append((A_temp+Lap_rf/((1/input_dim)**2)))
        return A_list
    else:
        print("Unknown input type!")
        return 0

def generate_data(task_list, data_dim, total_sep_num, cv_matrix=None, seed_value=100):
    np.random.seed(seed_value)
    if cv_matrix == None:
        cv_matrix = np.identity(data_dim)
        mean_arr = np.zeros(data_dim)

    task_num = len(task_list)
    input_all = []
    output_all = []
    for i in range(task_num):
        A_temp = task_list[i]
        A_temp_inv = np.linalg.inv(A_temp)
        input_temp = np.random.multivariate_normal(mean_arr, cv_matrix, total_sep_num) # size: total_num, d
        output_temp = np.einsum("ab,bc->ac", A_temp_inv, input_temp.T)
        input_all.append(input_temp)
        output_all.append(output_temp.T)
    return input_all, output_all

def get_data_normal(task_num, input_dim, total_sep_num, alpha=3, tau=5, seed_value=100):
    np.random.seed(seed_value)
    cv_matrix = np.identity(input_dim)
    mean_arr = np.zeros(input_dim)

    input_all = []
    output_all = []
    for i in range(task_num):
        input_temp = np.random.multivariate_normal(mean_arr, cv_matrix, total_sep_num) # size: total_num, d
    return input_temp

def generate_data_LN(task_list, data_dim, total_sep_num, alpha=3, tau=5, seed_value=100):
    np.random.seed(seed_value)
    task_num = len(task_list)
    input_all = []
    output_all = []
    grid_np = np.arange(data_dim) / data_dim
    for i in range(task_num):
        A_temp = task_list[i]
        A_temp_inv = np.linalg.inv(A_temp)
        input_list = []
        for j in range(total_sep_num):
            input_temp = KL_rf(grid_np, tau=tau, alpha=alpha, dim=1, order=20)
            input_list.append(input_temp)
        input_temp = np.array(input_list)
        #print(input_temp.shape)
        output_temp = np.einsum("ab,bc->ac", A_temp_inv, input_temp.T)
        input_all.append(input_temp)
        output_all.append(output_temp.T)
    return input_all, output_all

def generate_data_LN_fi(task_list, data_dim, total_sep_num, alpha=3, tau=5, seed_value=100):
    np.random.seed(seed_value)
    task_num = len(task_list)
    input_all = []
    output_all = []
    input_dim = task_list[0].shape[0]
    x_np, w_np = gauss_root_weight(2*input_dim+1,0,1)
    
    for i in range(task_num):
        A_temp = task_list[i]
        A_temp_inv = np.linalg.inv(A_temp)
        input_list = []
        #for j in range(total_sep_num):
        #    input_temp = KL_rf(x_np, tau=tau, alpha=alpha, dim=1, order=20)
        #    sin_input = np.array([np.sum(np.sqrt((1-0) / 2) * np.sqrt(2) * np.sin(i * np.pi * x_np) * w_np * input_temp) for i in range(1,input_dim+1)])
        #    input_list.append(sin_input)
        
        input_temp_temp = KL_rf_tensor(x_np, tau=tau, alpha=alpha, dim=total_sep_num, order=20)
        input_temp = np.einsum("ab,bc->ca", np.sqrt((1-0) / 2) * np.sqrt(2) *np.sin(np.einsum("ab,bc->ac",((np.arange(input_dim)+1)*np.pi).reshape(-1,1), x_np.reshape(1,-1))),np.einsum("a,ab->ab", w_np, input_temp_temp))
        
        #print(input_temp.shape)
        output_temp = np.einsum("ab,bc->ac", A_temp_inv, input_temp.T)
        input_all.append(input_temp)
        output_all.append(output_temp.T)
    return input_all, output_all

def generate_data_LN_fi_ne(task_list, data_dim, total_sep_num, scaler=1, alpha=1, tau=1, seed_value=100):
    np.random.seed(seed_value)
    task_num = len(task_list)
    input_all = []
    output_all = []
    input_dim = task_list[0].shape[0]
    x_np, w_np = gauss_root_weight(2*input_dim+1,0,1)
    
    for i in range(task_num):
        A_temp = task_list[i]
        A_temp_inv = np.linalg.inv(A_temp)
        input_list = []
        #for j in range(total_sep_num):
        #    input_temp = KL_rf(x_np, tau=tau, alpha=alpha, dim=1, order=20)
        #    sin_input = np.array([np.sum(np.sqrt((1-0) / 2) * np.sqrt(2) * np.sin(i * np.pi * x_np) * w_np * input_temp) for i in range(1,input_dim+1)])
        #    input_list.append(sin_input)
        
        input_temp_temp = KL_rf_tensor_ne(x_np, scaler=scaler, tau=tau, alpha=alpha, dim=total_sep_num, order=20)
        input_temp = np.einsum("ab,bc->ca", np.sqrt((1-0) / 2) * np.sqrt(2) *np.sin(np.einsum("ab,bc->ac",((np.arange(input_dim)+1)*np.pi).reshape(-1,1), x_np.reshape(1,-1))),np.einsum("a,ab->ab", w_np, input_temp_temp))
        
        #print(input_temp.shape)
        output_temp = np.einsum("ab,bc->ac", A_temp_inv, input_temp.T)
        input_all.append(input_temp)
        output_all.append(output_temp.T)
    return input_all, output_all
    
