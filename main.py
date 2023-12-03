# External imports
import numpy as np
import time
import matplotlib.pyplot as plt

# Own imports
from packages.func import func
import tensorflow as tf
from sklearn.model_selection import train_test_split

# # White noise
# mean = 0
# std = 1 
# num_samples = 1000
# samples = np.random.normal(mean, std, size=num_samples)

# t0 = time.time()
# cool_func = func("2 + 3*sin(x/2) + exp(x)^cos(x/3)")
# t1 = time.time()
# print(cool_func.eval(np.pi))
# t2 = time.time()

# print("Time to define: ", t1-t0, "s", sep="")
# print("Time to eval: ", t2-t1, "s", sep="")

# # Sympy
# import numpy as np
# import time
# from sympy import symbols, sin, exp, cos, lambdify

# t0 = time.time()
# x = symbols('x')
# expr = 2 + 3*sin(x/2) + exp(x)**cos(x/3)
# cool_func = lambdify(x, expr)
# t1 = time.time()
# print(cool_func(np.pi))
# t2 = time.time()

# print("Time to define: ", t1-t0, "s", sep="")
# print("Time to eval: ", t2-t1, "s", sep="")


# Defining basis functions:

#basis_functions = ['c', 'x', 'cos', 'exp']

def plot_sample(X):

    num_samples = 100
    start = -10
    end = 10
    step = (end - start)/num_samples
    eval_points = np.arange(start, end, step)

    plt.plot(eval_points,X)
    plt.show()

def generate_sample(f: func) -> (np.array, np.array):

    basis_function = f.get_value()
    y = np.zeros(len(basis_functions))
    for i in range(len(y)):
        if basis_functions[i] == basis_function:
            y[i] = 1

    # mean = np.random.uniform(-0.1, 0.1, 1)
    mean = 0
    std = np.random.uniform(0.1, 0.3, 1)

    num_samples = 100
    start = -10
    end = 10
    step = (end - start)/100
    eval_points = np.arange(start, end, step)

    X = []
    for point in eval_points:
        X.append(f.eval(point))

    noise = np.random.normal(mean, std, size=num_samples)
    X = X + noise

    return np.array(X), np.array(y)

def generate_samples(f_list: list[func], num_samples: int | list[int], shuffle: bool = True) -> (np.array, np.array):
    X_train = []
    y_train = []

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

    if shuffle:
        idx = np.random.permutation(len(X_train))
        return X_train[idx], y_train[idx]

    return X_train, y_train


f_list = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)', 'cos(x)']
basis_functions = []
for f in f_list:
    basis_functions.append(func(f))

X, y = generate_samples(basis_functions, 1000)


def vanilla_neural_network(input_shape, output_num):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_num, activation='softmax')
    ])

    return model

model = vanilla_neural_network(np.shape(X[0]), np.shape(y)[1])


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using X_train_new
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, steps_per_epoch=100)

# Test the model on X_test
# y_pred = model.predict(X_test)
f = func("(x*0.2)^5 + 6*sin(x)")
X_test, y_test = generate_samples([f], 20)
f.plot(-100, 100)
plot_sample(X_test[0])
y_pred = model.predict(X_test)
y_pred = model.predict(X_test)

print(np.around(y_pred, 3))

# Compare mean squared error of y_test and y_pred
count = 0
for i in range(len(y_test)):
    if np.argmax(y_test[i]) == np.argmax(y_pred[i]):
        count += 1

print("Number of rows where max(y_test)[row] == max(y_pred[row]):", count)
print("Number of rows where max(y_test)[row] != max(y_pred[row]):", len(y_test) - count)

plot_sample(X_test[0])

