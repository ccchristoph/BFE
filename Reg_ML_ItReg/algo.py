# External imports
import numpy as np
import time
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.model_selection import train_test_split

# Own imports
from ..packages.func import func

'''
Algorithm:
-1?) Pre-filter data, add normalzation (with highest value), etc. (only normalization should be necessary)
0) Start by evaluating which basis function in [1, x, x^2, x^3, sin(x), exp(x), ln(x)] (TODO: later add asin(x) etc as well?) fits best
1) After having found the initial guess, perform regression to find the best fit
2) Edit input data to ignore effects of initial guess. E.g. if the initial guess (plus regression) is 'y = a*sin(x) + b', then subtract 'b' from x, divide it by 'a' and perform asin(x). (TODO: implement limits - e.g. if bsis func is a*sin(x), then 'a' needs to be such that asin(x) can be taken. Same for ln(x) etc.)
3) Peform step 1 and 2 again (up to specified depth or until tolerance is fine)
4) Perform one last regression on each level, knowing what comes after (?)
end) inverse normalization
'''

# Training data
def generate_samples(f_or_list: list[func | str], num_samples: int | list[int], shift: float = 0, amplitude: float = 1, x_amp: float = 1, shuffle: bool = True) -> (np.array, np.array):
    basis_functions = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)']

    def generate_sample(f: func) -> (np.array, np.array):
        basis_function = f.get_value()
        y = np.zeros(len(basis_functions))
        for i in range(len(y)):
            if basis_functions[i] == basis_function:
                y[i] = 1

        # mean = np.random.uniform(-0.1, 0.1, 1)
        mean = 0
        std = np.random.uniform(0.1, 0.3, 1)

        num_pts = 100
        start = -10
        end = 10
        step = (end - start)/100
        eval_points = np.arange(start, end, step)

        X = np.array([])
        for point in eval_points:
            X = np.append(X, f.eval(point))

        noise = np.random.normal(mean, std, size=num_pts)
        X = X + noise

        return X, y
    
    X_train = []
    y_train = []

    f_list = f_or_list.copy() # Add error statement if f_or_list is not a list
    for f_ind, f in enumerate(f_list):
        if isinstance(f, str):
            f_list[f_ind] = func(f)
        elif not isinstance(f, func):
            raise TypeError("Functions passed in generate_samples must be of type func or str")

    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(f_list)

    for f_ind, f in enumerate(f_list):
        curr_num = num_samples[f_ind]
        for _ in range(curr_num):
            X, y = generate_sample(f)
            X_train.append(X)
            y_train.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("shape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    if shuffle:
        idx = np.random.permutation(len(X_train))
        return X_train[idx], y_train[idx]

    return X_train, y_train

print("Generating samples...")
X_train, y_train = generate_samples([func('sin(x)')], 10)
print("X_train: ", X_train)
print("y_train: ", y_train)