# Gradient Descent Assignment Solution

import numpy as np
import matplotlib.pyplot as plt


# Define the function f(x) and its derivative f'(x)
def f(x):
    return x ** 2 + 3 * x + 2


def gradient(x):
    return 2 * x + 3


# Implement gradient descent with a given starting point and learning rate
def gradient_descent(start_x, learning_rate, tolerance, max_iter=1000):
    x = start_x
    path = [x]

    for _ in range(max_iter):
        grad = gradient(x)
        if abs(grad) < tolerance:
            break
        x = x - learning_rate * grad
        path.append(x)

    return x, path


# Task 1: Gradient Descent with learning rate 0.1 and initial point 10
def task1():
    initial_x = 10
    learning_rate = 0.1
    tolerance = 1e-6
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    # Run gradient descent
    min_x, path = gradient_descent(initial_x, learning_rate, tolerance)
    path_y_vals = f(np.array(path))

    # Plot the descent path
    plt.plot(x_vals, y_vals, label='f(x) = x^2 + 3x + 2')
    plt.scatter(path, path_y_vals, color='red', label='Gradient Descent Path')
    plt.title('Gradient Descent on f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output minimum values
    print(f"Task 1: Minimum x = {min_x}, f(x) = {f(min_x)}")


# Task 2: Investigating the effect of different learning rates
def task2():
    learning_rates = [0.1, 0.5, 1.5]
    initial_x = 10
    tolerance = 1e-6
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    descent_paths = {}

    # Run gradient descent for each learning rate
    for lr in learning_rates:
        _, path = gradient_descent(initial_x, lr, tolerance)
        descent_paths[lr] = path

    # Plot the results
    plt.plot(x_vals, y_vals, label='f(x) = x^2 + 3x + 2')
    for lr, path in descent_paths.items():
        path_y_vals = f(np.array(path))
        plt.scatter(path, path_y_vals, label=f'Path with η = {lr}')

    plt.title('Gradient Descent with Different Learning Rates')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output minimum values for each learning rate
    for lr, path in descent_paths.items():
        min_x = path[-1]
        print(f"Task 2: Learning rate {lr}, Minimum x = {min_x}, f(x) = {f(min_x)}")


# Task 3: Exploring different starting points
def task3():
    initial_points = [10, -5, 0]
    learning_rate = 0.1
    tolerance = 1e-6
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    descent_paths = {}

    # Run gradient descent for each initial point
    for initial_x in initial_points:
        _, path = gradient_descent(initial_x, learning_rate, tolerance)
        descent_paths[initial_x] = path

    # Plot the results
    plt.plot(x_vals, y_vals, label='f(x) = x^2 + 3x + 2')
    for initial_x, path in descent_paths.items():
        path_y_vals = f(np.array(path))
        plt.scatter(path, path_y_vals, label=f'Path from x0 = {initial_x}')

    plt.title('Gradient Descent from Different Starting Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output results for each starting point
    for initial_x, path in descent_paths.items():
        min_x = path[-1]
        iterations = len(path) - 1
        print(f"Task 3: Starting point {initial_x}, Minimum x = {min_x}, f(x) = {f(min_x)}, Iterations = {iterations}")


# Task 4: Stability analysis
def task4():
    critical_point = -1.5
    perturbations = [0.01, 0.1, -0.01, -0.1]
    learning_rate = 0.1
    tolerance = 1e-6
    x_vals = np.linspace(-3, 3, 400)
    y_vals = f(x_vals)

    perturbed_paths = {}

    # Run gradient descent for perturbed points
    for delta in perturbations:
        perturbed_x0 = critical_point + delta
        _, path = gradient_descent(perturbed_x0, learning_rate, tolerance)
        perturbed_paths[delta] = path

    # Plot the results
    plt.plot(x_vals, y_vals, label='f(x) = x^2 + 3x + 2')
    for delta, path in perturbed_paths.items():
        path_y_vals = f(np.array(path))
        plt.scatter(path, path_y_vals, label=f'Perturbation δ = {delta}')

    plt.title('Stability Analysis: Gradient Descent with Perturbations')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output results for each perturbation
    for delta, path in perturbed_paths.items():
        final_x = path[-1]
        print(
            f"Task 4: Perturbation δ = {delta}: Converged to x = {final_x}, f(x) = {f(final_x)}, Iterations = {len(path) - 1}")


# Combining all tasks
def main():
    print("\nExecuting Task 1:")
    task1()

    print("\nExecuting Task 2:")
    task2()

    print("\nExecuting Task 3:")
    task3()

    print("\nExecuting Task 4:")
    task4()


# Run all tasks
main()
