import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time


def polynomial_fun(w, x):
    '''
    Implement a polynomial function polynomial_fun, that takes two input arguments, 
    a weight vector w of size M+1 and an input scalar variable x, and returns the function value y

    Args:
    w (Torch.tensor): weight vector with size M+1
    x (Torch.tensor): scalar variable

    Return:
    y (Torch.tensor): scalar
    '''

    # generate power value
    p = torch.arange(len(w))

    # reshape x -> (n, M+1)
    x_vec = torch.pow(x.unsqueeze(1), p) 
    
    # w0 x^0 + w1 x^1 + ... + wM+1 x^M+1
    y = torch.matmul(w, x_vec.t())

    return y.squeeze()

def fit_polynomial_ls(x, t, M):
    '''
    Implement a least square solver for fitting the polynomial functions, which takes N pairs of x and target values t as input, 
    with an additional input argument to specify the polynomial degree M, and returns the optimum weight vector 𝐰̂ in least-square sense

    Args:
    X (Torch.tensor): Nx1 vector
    t (Torch.tensor): Nx1 target values
    M (int): polynomial degree

    Return:
    w (Torch.tensor): optimum weight vector
    '''
    # generate power value
    p = torch.arange(M+1)

    # reshape x from Nx1 to Nx(M+1)
    x_mat = pow(x.unsqueeze(1), p)

    w = (torch.inverse(x_mat.t() @ x_mat)) @ x_mat.t() @ t

    return w

def fit_polynomial_sgd(x, t, M, lr, batch_size):
    '''
    Implement a stochastic minibatch gradient descent algorithm for fitting the polynomial functions
    This function also returns the optimum weight vector 𝐰


    Args:
    X (Torch.tensor): Nx1 vector
    t (Torch.tensor): Nx1 target values
    M (int): polynomial degree
    lr (float): learning rate
    batch_size (int): size of minibatch

    Return:
    w (torch.tensor): optimum weight vector
    '''
    
    num_epochs = 400
    num_batches = int(np.ceil(len(x) / batch_size))

    # Initialize w
    w = torch.randn(M+1, requires_grad=True)

    loss_fc = nn.MSELoss()
    optimizer = optim.SGD([w], lr=lr)

    # loader = DataLoader((x, t), shuffle=True, batch_size=batch_size, num_workers=0)
    print('---------------------------SGD training process--------------------------')

    for epoch in range(num_epochs):
        epoch_loss = 0
        indices = torch.randperm(len(x))

        for batch in range(num_batches):
            # Get mini-batch indices
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, len(x))
            batch_idx = indices[start_idx:end_idx]

            x_batch = x[batch_idx]
            t_batch = t[batch_idx]

            y_pred = polynomial_fun(w, x_batch)
            loss = loss_fc(y_pred, t_batch)

            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss = epoch_loss / num_batches

        # Print the loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch+1}, Loss: {epoch_loss.item():.4f}")
            
    return w

def predict_y(w, x, y):
    '''
    Calculate the predicted y, mean difference (and standard deviation) between predicted value and ground truth

    Args:
    w: weight vector
    '''
    y_pred = polynomial_fun(w, x)

    mean_diff = torch.mean(y - y_pred)
    std = torch.std(y - y_pred)

    return mean_diff, std, y_pred

def generate_data():
    '''
    Generate w, training set, testing set and whole y
    '''
    # Initialize weight vector
    w = torch.tensor([1., 2., 3., 4., 5.])

    # Generate train set and test set
    x_train = 40 * torch.rand(100) - 20
    x_test = 40 * torch.rand(50) - 20
    
    # Using train set and test set to calculate ground truth value
    y_train = polynomial_fun(w, x_train)
    y_test = polynomial_fun(w, x_test)

    x = torch.cat((x_train, x_test))
    y = torch.cat((y_train, y_test))

    return w, x_train, x_test, y_train, y_test, x, y

if __name__ == '__main__':
    w, x_train, x_test, y_train, y_test, x, y = generate_data()

    # Calculate observed training data by adding guassian noise
    mean = torch.zeros(len(y_train))
    std = torch.ones(len(y_train)) * 0.2
    t = y_train + torch.normal(mean=mean, std=std)

    # least-square
    start = time.time()
    w_pred = fit_polynomial_ls(x_train, t, M=5)
    end =time.time()
    time_ls = end - start

    # predicted value
    y_pred = polynomial_fun(w_pred, x)

    # a) 
    print('------------Between observed training data and ground truth------------')
    print(f'| Mean difference: {torch.mean(t - y_train)} \t| Standard deviation: {torch.std(t - y_train)}')

    # b) 
    print('\n')
    print('------------Between LS-predicted value and ground truth------------')
    print(f'| Mean difference: {torch.mean(y - y_pred)} \t| Standard deviation: {torch.std(y - y_pred)}')
    print('\n')

    # Stochastic minibatch gradient descent
    start = time.time()
    w_sgd = fit_polynomial_sgd(x_train, t, M=5, lr=3e-13, batch_size=10)
    end = time.time()
    time_sgd = end - start

    # SGD-predicted value
    y_sgd = polynomial_fun(w_sgd, x)

    print('\n')
    print('------------Between SGD-predicted value and ground truth------------')
    print(f'| Mean difference: {torch.mean(y - y_sgd)} \t| Standard deviation: {torch.std(y - y_sgd)}')
    print('\n')

    # report the RMSEs in both w and y
    w_gt = torch.cat((w, torch.tensor([0]))) # To fit M=5 in the above questions

    rmse_w_ls = torch.sqrt(torch.mean((w_gt - w_pred) ** 2))
    rmse_w_sgd = torch.sqrt(torch.mean((w_gt - w_sgd) ** 2))
    rmse_y_ls = torch.sqrt(torch.mean((y - y_pred) ** 2))
    rmse_y_sgd = torch.sqrt(torch.mean((y - y_sgd) ** 2))


    print('---------------------------------------------------------------------------------------------')
    print('| \t\t |\t RMSE of w \t|\t RMSE of y \t| time spent in training |')
    print('---------------------------------------------------------------------------------------------')
    print('| Least Squares |\t %.4f \t|\t %.4f \t|\t %.4f [s] \t|' % (rmse_w_ls, rmse_y_ls, time_ls) )
    print('|\t SGD \t|\t %.4f \t|\t %.4f \t|\t %.4f [s] \t|' % (rmse_w_sgd, rmse_y_sgd, time_sgd))
    print('---------------------------------------------------------------------------------------------')
    print('\n')

