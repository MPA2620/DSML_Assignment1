import numpy as np
import matplotlib.pyplot as plt


# Define the function f(x1, x2) and its gradient
def f(x1, x2):
    return x1 ** 2 + x2 ** 2 + 3 * x1 + 4 * x2 + 5


def gradient(x1, x2):
    grad_x1 = 2 * x1 + 3
    grad_x2 = 2 * x2 + 4
    return np.array([grad_x1, grad_x2])


# Implement gradient descent for 2D function
def gradient_descent_2d(learning_rate, initial_x, tolerance, max_iter=1000):
    x = np.array(initial_x)  # Initial point (x1, x2)
    path = [x]

    for i in range(max_iter):
        grad = gradient(x[0], x[1])
        if np.linalg.norm(grad) < tolerance:
            break
        x = x - learning_rate * grad
        path.append(x)

    return np.array(path)


# Task 1: Gradient Descent with initial point (5, 5)
def task1():
    learning_rate = 0.1
    initial_x = [5, 5]  # Starting point
    tolerance = 1e-6

    # Run gradient descent
    path = gradient_descent_2d(learning_rate, initial_x, tolerance)

    # Generate contour plot
    x1_vals = np.linspace(-10, 10, 400)
    x2_vals = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)

    plt.contour(X1, X2, Z, levels=50)  # Contour plot
    plt.plot(path[:, 0], path[:, 1], 'r.-', label='Gradient Descent Path')  # Gradient descent path
    plt.scatter(path[-1, 0], path[-1, 1], color='red', label='Minimum')
    plt.title('Task 1: Gradient Descent in 2D')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the minimum point
    min_x1, min_x2 = path[-1]
    print(f"Task 1: Minimum found at x1 = {min_x1}, x2 = {min_x2}, f(x1, x2) = {f(min_x1, min_x2)}")


# Task 2: Investigating different learning rates
def task2():
    learning_rates = [0.1, 0.5, 1.0]
    initial_x = [5, 5]  # Starting point
    tolerance = 1e-6

    # Generate contour plot
    x1_vals = np.linspace(-10, 10, 400)
    x2_vals = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)

    plt.contour(X1, X2, Z, levels=50)  # Contour plot

    # Run gradient descent for each learning rate and plot the path
    for eta in learning_rates:
        path = gradient_descent_2d(eta, initial_x, tolerance)
        plt.plot(path[:, 0], path[:, 1], label=f'Path with η = {eta}', marker='o')

    # Customize and display the plot
    plt.scatter(path[-1, 0], path[-1, 1], color='red', label='Final Minimum')
    plt.title('Task 2: Gradient Descent with Different Learning Rates')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the final results for each learning rate
    for eta in learning_rates:
        path = gradient_descent_2d(eta, initial_x, tolerance)
        min_x1, min_x2 = path[-1]
        print(
            f"Task 2: Learning rate: {eta}, Minimum found at x1 = {min_x1}, x2 = {min_x2}, f(x1, x2) = {f(min_x1, min_x2)}")


# Task 3: Exploring different starting points
def task3():
    learning_rate = 0.1
    initial_points = [(5, 5), (-5, 5), (-5, -5)]  # Different starting points
    tolerance = 1e-6

    # Generate contour plot
    x1_vals = np.linspace(-10, 10, 400)
    x2_vals = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)

    plt.contour(X1, X2, Z, levels=50)  # Contour plot

    # Run gradient descent for each starting point and plot the path
    for initial_x in initial_points:
        path = gradient_descent_2d(learning_rate, initial_x, tolerance)
        plt.plot(path[:, 0], path[:, 1], label=f'Path from {initial_x}', marker='o')

    # Customize and display the plot
    plt.scatter(path[-1, 0], path[-1, 1], color='red', label='Final Minimum')
    plt.title('Task 3: Gradient Descent from Different Starting Points')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the final results for each starting point
    for initial_x in initial_points:
        path = gradient_descent_2d(learning_rate, initial_x, tolerance)
        min_x1, min_x2 = path[-1]
        print(
            f"Task 3: Starting point: {initial_x}, Minimum found at x1 = {min_x1}, x2 = {min_x2}, f(x1, x2) = {f(min_x1, min_x2)}")


# Task 4: Stability analysis with perturbations around the critical point
def task4():
    critical_point = (-1.5, -2)
    print(
        f"Task 4: Critical point: x1 = {critical_point[0]}, x2 = {critical_point[1]}, f(x1, x2) = {f(*critical_point)}")

    perturbations = [(0.1, 0.1), (-0.1, -0.1), (0.05, -0.05)]  # Small perturbations
    learning_rate = 0.1
    tolerance = 1e-6

    # Generate contour plot
    x1_vals = np.linspace(-3, 1, 400)
    x2_vals = np.linspace(-3.5, 1, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)

    plt.contour(X1, X2, Z, levels=50)  # Contour plot

    # Run gradient descent for each perturbed starting point and plot the path
    for delta in perturbations:
        perturbed_x = (critical_point[0] + delta[0], critical_point[1] + delta[1])
        path = gradient_descent_2d(learning_rate, perturbed_x, tolerance)
        plt.plot(path[:, 0], path[:, 1], label=f'Perturbation {delta}', marker='o')

    # Customize and display the plot
    plt.scatter(critical_point[0], critical_point[1], color='red', label='Critical Point')
    plt.title('Task 4: Gradient Descent Stability Near Critical Point')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the results for each perturbation
    for delta in perturbations:
        perturbed_x = (critical_point[0] + delta[0], critical_point[1] + delta[1])
        path = gradient_descent_2d(learning_rate, perturbed_x, tolerance)
        min_x1, min_x2 = path[-1]
        print(
            f"Task 4: Perturbation: {delta}, Converged to x1 = {min_x1}, x2 = {min_x2}, f(x1, x2) = {f(min_x1, min_x2)}")


# Combine all tasks
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
