import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(x0, learning_rate, gradient_func, iterations):
    """
    Simulates the gradient descent update.

    Args:
    - x0: Initial point
    - learning_rate: Learning rate η
    - gradient_func: Function that returns the gradient ∇f(x)
    - iterations: Maximum number of iterations

    Return:
    - List of x values after each iteration (for visualization)
    """
    x = x0
    history = [x0]  # List to store the trajectory
    for i in range(iterations):
        grad = gradient_func(x)
        x = x - learning_rate * grad  # Gradient descent update rule
        history.append(x)

        # Stop if the gradient is small enough
        if abs(grad) < 1e-6:
            print(f"Convergence at iteration {i}")
            break
    return history


# Define the function f(x) = x^2 + 3x + 2 and its gradient
def f_prime(x):
    return 2 * x + 3  # Gradient of the function f(x)


# Example usage
x0 = 10  # Initial point
learning_rate = 0.1
iterations = 100

trajectory = gradient_descent(x0, learning_rate, f_prime, iterations)

# Plot the trajectory
x_vals = np.linspace(-10, 10, 400)
f_vals = x_vals ** 2 + 3 * x_vals + 2

plt.plot(x_vals, f_vals, label="f(x) = x^2 + 3x + 2")
plt.scatter(trajectory, [x ** 2 + 3 * x + 2 for x in trajectory], color='red', label="Gradient Descent Path")
plt.title("Gradient Descent Trajectory")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
