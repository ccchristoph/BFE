import sys
sys.path.append('/home/arturo/OwnCode/co-repos/BFE/')

# External imports
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Own imports
from packages.func import func

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

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(np.vstack((a, b)))
# print(np.hstack((a, b)))
# print(np.column_stack((a, b)))

if True:
    import tensorflow as tf
    # Import model
    model = tf.keras.models.load_model('Reg_ML_ItReg/model.keras') # Attention: When doing model.predict(), the input needs to be 2D (meaning for 1 array, it needs to be np.array([[...]]))
    # y_pred = model.predict(np.array([np.sin(np.arange(0, 10, 0.1))]))
    # print(y_pred[0]/np.max(y_pred[0]))

    # Example noisy sample - x^2 + sin(x) + noise
    x = np.linspace(-10, 10, 100)
    # y = (x-2.25)**2 + 5*np.sin(5*(x-2.25)) + np.random.normal(0, 0.15, len(x))
    y = (x-0)**2 + 5*np.sin(5*(x-0)) + np.random.normal(0, 0.15, len(x))

    def plot_samples(X: np.array, y: np.array, fun_range: tuple[float, float]):
        # TODO: Add check for y to be a np.array and not e.g. list
        basis_str = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)']
        if len(np.shape(X)) == 1:
            single = True
            n_fig = 1
            l = len(X)
        else:
            single = False
            n_fig = np.shape(X)[0]
            l = np.shape(X)[1]
        N = 30
        n_fig_per = min(N, n_fig)
        n_rows = int(np.floor(np.sqrt(n_fig_per)))
        n_cols = int(np.ceil(n_fig_per / n_rows))

        x_values = np.linspace(fun_range[0], fun_range[1], num=l)

        # Iterate over figures
        for figure in range(n_fig):
            # Create a new figure every N subplots
            if figure % N == 0:
                fig, axes = plt.subplots(n_rows, n_cols)
                axes = np.array(axes).flatten()

            # Plot the current array in a subplot
            ax = axes[figure % N]
            if single:
                ax.plot(x_values, X)
            else:
                ax.plot(x_values, X[figure])
            ax.set_xlim(fun_range)

            if single:
                y_curr = y
            else:
                y_curr = y[figure]

            func_index = np.where(y_curr == 1)
            print(func_index)
            if len(func_index[0]) > 1:
                raise ValueError("More than one function is set true in y during plotting")

            ax.set_title(basis_str[np.where(y_curr == 1)[0][0]])

            # If the last subplot of the current figure, show and save it
            if figure % N == N - 1 or figure == n_fig - 1:
                plt.draw()
                # Optionally, save the figure
                # fig.savefig(f'figure_{figure // N + 1}.png')

    # Algorithm
    def BFE(sample: np.array, fun_range: tuple[float, float], depth: int, tolerance: float = 0.01): # Sampel 1D for now (TODO: make it work for 2D as well)
        basis_str = ['0', 'x', 'x^2', 'x^3', 'sin(x)', 'exp(x)']
        basis_functions = []
        for f in basis_str:
            basis_functions.append(func(f))

        inv_basis_str = ['0', '1/x', 'x^(1/2)', 'x^(1/3)', 'asin(x)', 'ln(x)'] # TODO: Instead of depth, go until function found is constant (add emergeny depth of e.g. 100)
        inv_basis_functions = []
        for f in basis_str:
            # basis_functions.append(func(f)) # TODO: Expand func class with further functions (to have inverses)
            pass
        
        def ML_estimate(sample: np.array) -> list[tuple[func, func]] | None:
            y_pred = model.predict(np.array([sample]))[0] # TODO: make it work for 2D as well
            print(np.around(y_pred/max(y_pred), 3))
            poss_fun = np.array(basis_functions)[y_pred > 0.3]
            if len(poss_fun) == 0:
                raise ValueError("TODO: End recursion on this tree here")
                return None
            else:
                return [poss_fun, poss_fun] # TODO: Replace second element with inverse
            
        def regression(sample: np.array, fun_range: tuple[float, float], f: func) -> np.array: # TODO: make it work for 2D as well
            X_reg = np.column_stack((np.ones_like(sample), f.eval(np.linspace(fun_range[0], fun_range[1], len(sample)))))
            y_reg = sample
            reg = LinearRegression()
            reg.fit(X_reg, y)
            coefficients = reg.coef_
            return coefficients
            # return func(f'{coefficients[1]}*{f.root.value} + {coefficients[0]}')
        
        def transform_data(sample: np.array, coefficients: np.array, inverse: func) -> np.array: # TODO: make it work for 2D as well
            return inverse.eval((sample - coefficients[0])/coefficients[1])
            
        est = ML_estimate(sample)
        for f in est:
            print(f.root.value)
            coeffs = regression(sample, fun_range, f)
            plot_samples(transform_data(sample, coeffs), np.array([0, 0, 1, 0, 0, 0]), (-10, 10))
            # print(regression(sample, fun_range, f).root.value)

    BFE(y, (-10, 10), 1)

    # plot_samples(y, np.array([0, 0, 1, 0, 0, 0]), (-10, 10))

    plt.show()
