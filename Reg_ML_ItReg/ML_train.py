import sys
sys.path.append('/home/arturo/OwnCode/co-repos/BFE/')

# External imports
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import sympy as sp

# Own imports
from packages.func import func

# No blocking of program when plotting - use the Qt5Agg backend for non-blocking behavior
# plt.switch_backend('Qt5Agg')
# plt.ion()
# plt.show(block=False)

'''
Training of Data:
- Create multiple datasamples, possibly with varying amplitudes and shifts, then (possibly normalize) train an ML model on it - outputs should be a vec of probabilities (using softmax activation function).
- Train the model and save it to be used later
'''

# Training data
def generate_samples(num_samples: int | list[int], num_pts: float, fun_range: tuple[float, float], x_shift: float = 0, x_amplitude: float = 1, y_shift: float = 0, y_amplitude: float = 1, shuffle: bool = True, normalize: bool = True, filter: bool = True) -> (np.array, np.array): # Change amplitudes and shifts to kwargs, which's values should be tuples (bool random, float value)
    basis_str = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)'] # Add (also in plotting): x^(-1), x^(-2), ln(x), asin(x), acos(x), atan(x), sinh(x), cosh(x), tanh(x), asinh(x), acosh(x), atanh(x)
    basis_functions = []
    for f in basis_str:
        basis_functions.append(func(f))

    def generate_sample(f: func, num_pts: float, fun_range: tuple[float, float], x_shift: float, x_amplitude: float, y_shift: float, y_amplitude: float) -> (np.array, np.array):
        # Fill y
        function_value = f.get_value()
        y = np.zeros(len(basis_functions))
        for i in range(len(y)):
            if function_value == basis_functions[i].get_value():
                y[i] = 1

        # shift_x and x-amplitude
        x_sh = np.random.uniform(-x_shift, x_shift)
        x_amp = np.random.uniform(1/x_amplitude, x_amplitude)
        f_changed = func(f.get_value())
        f_changed.subs_x(f'{x_amp}*x+{x_sh}')
        # shift_y and y-amplitude
        if f.get_value() != 'x':
            y_sh = np.random.uniform(-y_shift, y_shift)
            y_amp = np.random.uniform(1/y_amplitude, y_amplitude)
            f_changed = func(str(y_amp) + '*(' + f_changed.get_value() + ')+' + str(y_sh))

        # print("f_changed: ", f_changed.get_value())

        # Evaluation points
        start = fun_range[0]
        end = fun_range[1]
        if start >= end:
            raise ValueError("Start of fun_range is larger than or equal to end")
        step = (end - start)/num_pts
        eval_points = np.arange(start, end, step)

        X = np.array([])
        for point in eval_points:
            X = np.append(X, f_changed.eval(point))

        if normalize:
            X = X/np.max(X) # Normalization
        
        # y-noise
        mean = 0
        std = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(mean, std, size=num_pts)
        X += noise
        # plot_samples([X], [y], fun_range)

        if filter:
            X = apply_mean_filter(X, int(np.cbrt(num_pts)))

        return X, y
    
    X_train = []
    y_train = []

    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(basis_functions)

    for f_ind, f in enumerate(basis_functions):
        curr_num = num_samples[f_ind]
        for _ in range(curr_num):
            X, y = generate_sample(f, num_pts, fun_range, x_shift, x_amplitude, y_shift, y_amplitude)
            X_train.append(X)
            y_train.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if shuffle:
        idx = np.random.permutation(len(X_train))
        return X_train[idx], y_train[idx]

    return X_train, y_train

# def plot_samples(X: np.array, y: np.array, fun_range: tuple[float, float]):
#     basis_str = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)']
#     n_fig = np.shape(X)[0]
#     n_rows = int(np.floor(np.sqrt(n_fig)))
#     n_cols = int(np.ceil(n_fig / n_rows))
#     fig, axes = plt.subplots(n_rows, n_cols)
#     axes = axes.flatten()

#     # # Title of figure
#     # fig.suptitle(f'Class {class_ind}')

#     x_values = np.linspace(fun_range[0], fun_range[1], num=len(X[0]))
#     # Plot each array in a subplot
#     for figure, ax in zip(range(n_fig), axes):
#         ax.plot(x_values, X[figure])
#         ax.set_xlim(fun_range)
#         y_curr = y[figure]
#         func_index = np.where(y_curr == 1)
#         if len(func_index) > 1:
#             raise ValueError("More than one function is set true in y during plotting")
#         ax.set_title(basis_str[np.where(y_curr == 1)[0][0]])
#     plt.draw()

def plot_samples(X: np.array, y: np.array, fun_range: tuple[float, float]):
    basis_str = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)']
    n_fig = np.shape(X)[0]
    N = 30
    n_fig_per = min(N, n_fig)
    n_rows = int(np.floor(np.sqrt(n_fig_per)))
    n_cols = int(np.ceil(n_fig_per / n_rows))

    x_values = np.linspace(fun_range[0], fun_range[1], num=len(X[0]))

    # Iterate over figures
    for figure in range(n_fig):
        # Create a new figure every N subplots
        if figure % N == 0:
            fig, axes = plt.subplots(n_rows, n_cols)
            axes = np.array(axes).flatten()

        # Plot the current array in a subplot
        ax = axes[figure % N]
        ax.plot(x_values, X[figure])
        ax.set_xlim(fun_range)

        y_curr = y[figure]
        func_index = np.where(y_curr == 1)
        if len(func_index) > 1:
            raise ValueError("More than one function is set true in y during plotting")

        ax.set_title(basis_str[np.where(y_curr == 1)[0][0]])

        # If the last subplot of the current figure, show and save it
        if figure % N == N - 1 or figure == n_fig - 1:
            plt.draw()
            # Optionally, save the figure
            # fig.savefig(f'figure_{figure // N + 1}.png')

def apply_mean_filter(data:np.array , kernel_size: int):
    # Define the kernel for the mean filter
    kernel = np.ones(kernel_size) / kernel_size

    # Apply the mean filter using numpy.convolve
    smoothed_data = np.convolve(data, kernel, mode='same')

    return smoothed_data

def find_nth_largest(arr: np.array, n: int):
    if n > len(arr):
        raise ValueError("n is larger than the length of arr")
    np_arr = np.copy(arr)
    return np.argsort(np_arr)[-2]

def vanilla_neural_network(input_shape, output_num):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_num, activation='softmax')
    ])

    return model

normalize = False
filter = True
fun_range = (-10, 10)
basis_str = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)']
X_train, y_train = generate_samples(1000, 100, fun_range, x_shift=0.5, x_amplitude=0.5, y_shift=0.5, y_amplitude=0.5, normalize=normalize, filter=filter)
# plot_samples(X_train, y_train, fun_range)

### Train model
model = vanilla_neural_network(np.shape(X_train[0]), np.shape(y_train)[1])

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model using X_train_new
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, steps_per_epoch=100)

### Test prediction (possibly on new data)
num_new_data = 1000
if num_new_data > 0:
    X_test, y_test = generate_samples(num_new_data, 100, fun_range, x_shift=0.5, x_amplitude=0.5, y_shift=0.5, y_amplitude=0.5, normalize=normalize, filter=(not filter))
y_pred = model.predict(X_test)
y_pred_index = np.argmax(y_pred, axis=1)
y_pred_str = []
for i in range(len(y_pred_index)):
    y_pred_str.append(basis_str[y_pred_index[i]])

# print(y_pred_str)
# plot_samples(X_test, y_test, fun_range)

# Compare mean squared error of y_test and y_pred

count = 0
for i in range(len(y_test)):
    true_ind = np.argmax(y_test[i])
    # print("True index:", true_ind)
    # print("Type of true index:", type(true_ind))
    # if np.argmax(y_test[i]) == np.argmax(y_pred[i]) or find_nth_largest(y_test[i], 2) == np.argmax(y_pred[i]) or find_nth_largest(y_test[i], 3) == np.argmax(y_pred[i]):
    # if true_ind == np.argmax(y_pred[i]):
    #     count += 1
    if y_pred[i][true_ind] > 0.4: # TODO: Later take all values bigger than 0.3 to test out with regression
        count += 1
    else:
        print("\nHighest probability:", np.max(y_pred[i]))
        print("Predicted function:", basis_str[np.argmax(y_pred[i])])
        print("Real function:", basis_str[np.argmax(y_test[i])])
        print("Probability of real function:", y_pred[i][true_ind])
    # else:
    #     print("Highest probability:", np.max(y_pred[i]))
    #     if np.max(y_pred[i]) > 0.8:
    #         print("Real function:", basis_str[np.argmax(y_test[i])])
    #         print("Predicted function:", basis_str[np.argmax(y_pred[i])])

print("Number of rows where max(y_test)[row] == max(y_pred[row]):", count)
print("Number of rows where max(y_test)[row] != max(y_pred[row]):", len(y_test) - count)

model.save('Reg_ML_ItReg/model.keras')

# Last line of script to keep plots open
plt.show()