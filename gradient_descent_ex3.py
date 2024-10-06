import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---
def mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

# --- Task 1: Single Variable Linear Regression using Gradient Descent ---
def linear_regression_1d(X, Y, learning_rate, tolerance, max_iter=1000):
    # Initialize weights and bias
    w, b = 0.0, 0.0
    n = len(Y)
    path = []

    # Perform gradient descent
    for _ in range(max_iter):
        y_pred = w * X + b
        # Calculate gradients
        grad_w = (-2/n) * sum(X * (Y - y_pred))
        grad_b = (-2/n) * sum(Y - y_pred)
        # Store current weights and bias
        path.append((w, b))
        # Update weights and bias
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Check if convergence criterion is met
        if abs(grad_w) < tolerance and abs(grad_b) < tolerance:
            break

    return w, b, path

# --- Task 2: Multiple Variable Linear Regression using Gradient Descent ---
def linear_regression_2d(X1, X2, Y, learning_rate, tolerance, max_iter=1000):
    # Initialize weights and bias
    w1, w2, b = 0.0, 0.0, 0.0
    n = len(Y)
    path = []

    # Perform gradient descent
    for _ in range(max_iter):
        y_pred = w1 * X1 + w2 * X2 + b
        # Calculate gradients
        grad_w1 = (-2/n) * sum(X1 * (Y - y_pred))
        grad_w2 = (-2/n) * sum(X2 * (Y - y_pred))
        grad_b = (-2/n) * sum(Y - y_pred)
        # Store current weights and bias
        path.append((w1, w2, b))
        # Update weights and bias
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        b -= learning_rate * grad_b

        # Check if convergence criterion is met
        if abs(grad_w1) < tolerance and abs(grad_w2) < tolerance and abs(grad_b) < tolerance:
            break

    return w1, w2, b, path

# --- Visualization for 1D Regression ---
def plot_1d_regression(X, Y, w, b, title="Linear Regression with Gradient Descent"):
    plt.scatter(X, Y, color='blue', label='Data Points')
    y_pred = w * X + b
    plt.plot(X, y_pred, color='red', label=f'Fitted Line: y = {w:.2f}x + {b:.2f}')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Visualization for 2D Regression ---
def plot_2d_regression(X1, X2, Y, w1, w2, b, title="Multiple Linear Regression with Gradient Descent"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, Y, color='blue', label='Data Points')

    # Create a grid for the surface plot
    X1_grid, X2_grid = np.meshgrid(np.linspace(min(X1), max(X1), 20),
                                   np.linspace(min(X2), max(X2), 20))
    Y_pred_grid = w1 * X1_grid + w2 * X2_grid + b

    # Use plot_surface instead of plot_trisurf
    ax.plot_surface(X1_grid, X2_grid, Y_pred_grid, color='red', alpha=0.5)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_title(title)
    plt.legend()
    plt.show()

# --- Main Functions for Task Execution ---
def task1():
    # Dataset 1
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2, 4, 6, 8, 10])
    learning_rate = 0.01
    tolerance = 1e-6
    w, b, path = linear_regression_1d(X, Y, learning_rate, tolerance)
    print(f"Task 1: Final Parameters: w = {w:.6f}, b = {b:.6f}, MSE = {mse(Y, w * X + b):.6f}")
    plot_1d_regression(X, Y, w, b)

def task2():
    # Dataset 2
    X1 = np.array([1, 2, 3, 4, 5])
    X2 = np.array([2, 3, 4, 5, 6])
    Y = np.array([2, 3, 5, 6, 8])
    learning_rate = 0.01
    tolerance = 1e-6
    w1, w2, b, path = linear_regression_2d(X1, X2, Y, learning_rate, tolerance)
    print(f"Task 2: Final Parameters: w1 = {w1:.6f}, w2 = {w2:.6f}, b = {b:.6f}, MSE = {mse(Y, w1 * X1 + w2 * X2 + b):.6f}")
    plot_2d_regression(X1, X2, Y, w1, w2, b)

# --- Main Menu for Task Selection ---
def main():
    while True:
        print("\nSelect the task to run:")
        print("1 - Task 1: Single Variable Linear Regression")
        print("2 - Task 2: Multiple Variable Linear Regression")
        print("0 - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            task1()
        elif choice == '2':
            task2()
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please select again.")

# Run the main menu
if __name__ == "__main__":
    main()
