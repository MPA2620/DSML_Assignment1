# DSML_Assignment1
First Assignment in DSML Courses
---

## Gradient Descent Exercise 1 - Explanation and Implementation

### Overview
The purpose of this exercise is to implement and explore gradient descent in 1D to minimize a quadratic function, analyze the effect of learning rates, starting points, and perform stability analysis around the minimum.

### Function:
The function given is:


      f(x) = x^2 + 3x + 2


It's a simple quadratic function with a minimum that can be found analytically as well as using gradient descent. The goal is to perform gradient descent to minimize this function and investigate how the parameters like learning rate and initial points affect the behavior of the algorithm.

---

### **Task 1: Gradient Descent for Minimization**

#### Goal:
Perform gradient descent with a learning rate of (eta = 0.1) starting from (x_0 = 10) and iterating until the gradient's absolute value is less than (10^{-6}).

#### Thought Process:
1. We defined the function f(x) and its derivative (f'(x) = 2x + 3), which is needed for gradient descent.
2. The gradient descent loop was set to update x iteratively
   until convergence is reached (when the gradient is small).
3. We plotted the path of the descent to visually observe the approach to the minimum.

---

### **Task 2: Effect of Different Learning Rates**

#### Goal:
Investigate the behavior of gradient descent using three different learning rates: (eta = 0.1), (eta = 0.5), and (eta = 1.5).

#### Thought Process:
1. We ran the same gradient descent algorithm as in Task 1 but experimented with different learning rates to observe how they affect convergence.
2. **Small learning rates** (eta = 0.1) lead to slow but steady convergence.
3. **Medium learning rates** (eta = 0.5) showed faster convergence.
4. **Large learning rates** (eta = 1.5) caused the algorithm to overshoot the minimum, resulting in divergence and overflow errors. This behavior is expected for high learning rates.
5. We handled the overflow errors using exception handling, but left the core algorithm intact to showcase the sensitivity of gradient descent to large learning rates.

---

### **Task 3: Effect of Different Starting Points**

#### Goal:
Explore the behavior of gradient descent when starting from different initial points: (x_0 = 10), (x_0 = -5), and (x_0 = 0).

#### Thought Process:
1. Different initial points were tested to understand how far from the minimum the starting point is and how it affects the convergence.
2. As expected, starting points closer to the minimum converged faster, while farther starting points required more iterations.
3. We plotted the paths for each initial point and compared their behavior. This task demonstrates how initial conditions can influence the speed and path of gradient descent.

---

### **Task 4: Stability Analysis**

#### Goal:
Perform a basic stability analysis by examining the behavior of gradient descent near the minimum and after small perturbations around the minimum \( x = -1.5 \).

#### Thought Process:
1. We analytically found the minimum by solving (f'(x) = 0), which gives (x = -1.5).
2. Small perturbations were introduced around this minimum, and gradient descent was applied to see how the algorithm behaves.
3. For small perturbations, the algorithm quickly returns to the minimum, showcasing stability around (x = -1.5).
4. This task emphasizes the importance of stability analysis in gradient-based optimization methods.

---

### Key Lessons Learned:
- **Learning Rate Sensitivity**: Large learning rates can lead to divergence, while small rates ensure convergence but may be slow.
- **Initial Conditions**: The choice of initial points significantly affects the number of iterations needed to reach the minimum.
- **Stability**: Near the critical point, small perturbations are corrected by the gradient descent, demonstrating its stability when close to the minimum.


Here’s a detailed explanation of **Exercise 2: Gradient Descent in 2D**, covering the tasks, implementation process, and the thought process behind each solution. This can be used as a documentation file to accompany your code.

---

## Gradient Descent in 2D - Explanation and Thought Process

### Overview
This exercise involves minimizing a **2D quadratic function** using gradient descent and analyzing the effects of learning rates, starting points, and stability near the critical point. Gradient descent is a fundamental optimization algorithm used extensively in machine learning for minimizing cost functions.

The function to minimize is:

      f(x_1, x_2) = x_1^2 + x_2^2 + 3x_1 + 4x_2 + 5

This gradient is used to iteratively update the values of (x_1) and (x_2) in gradient descent.

---

### **Task 1: Implement Gradient Descent**

#### Goal:
- Write a Python function to perform gradient descent on the 2D function.
- Set the learning rate (eta = 0.1) and start at the initial point (5, 5).
- Plot the path of gradient descent on the contour plot of the function.

### **Task 2: Effect of Different Learning Rates**

#### Goal:
- Investigate the effect of different learning rates (eta = 0.1, 0.5, 1.0) on the behavior of gradient descent.
- Plot the paths for each learning rate on the contour plot and compare their behavior.

#### Thought Process:
1. **Learning Rates**:
   - **Small Learning Rate** (eta = 0.1): This leads to slow convergence but steady, accurate results.
   - **Medium Learning Rate** (eta = 0.5): This should speed up convergence but may result in slight overshooting.
   - **Large Learning Rate** (eta = 1.0): A larger learning rate may cause the algorithm to overshoot the minimum or even diverge.
2. **Path Comparison**: By running gradient descent for each learning rate and plotting the paths, we were able to observe how the learning rates affected the speed and accuracy of convergence.
3. **Conclusion**: The smaller learning rates converged steadily, while the larger learning rate (eta = 1.0) may result in overshooting and instability.

---

### **Task 3: Effect of Different Starting Points**

#### Goal:
- Explore the effect of different initial points on gradient descent.
- Use starting points (5, 5), (-5, 5), and (-5, -5), and compare the paths taken to the minimum.

#### Thought Process:
1. **Initial Points**: We selected three different starting points to understand how the location of the initial point affects the convergence path and speed. The further from the minimum, the longer the algorithm might take to converge.
2. **Path Comparison**: Each starting point leads to a different path, but all paths should converge to the same minimum. The path lengths and directions, however, vary depending on how far the initial point is from the minimum.
3. **Conclusion**: The starting point significantly influences the number of iterations and the path taken, but all points eventually converge to the same minimum.

---

### **Task 4: Stability Analysis**

#### Goal:
- Perform a basic stability analysis by solving and analyzing the stability of the critical point.
- Introduce small perturbations around the minimum and observe the behavior of gradient descent near this point.

---

## Gradient Descent Exercise 3 - Explanation and Implementation

This project implements linear regression models using the Gradient Descent optimization algorithm in Python. The code is structured to handle both **single-variable linear regression** and **multiple-variable linear regression**. The project also includes data visualization for regression lines and 3D surfaces.

---

# Project Structure

The project consists of the following main components:

1. **Helper Functions**:
    - `mse(y_true, y_pred)`: Calculates the Mean Squared Error (MSE) between actual and predicted values.

2. **Gradient Descent Implementations**:
    - `linear_regression_1d(X, Y, learning_rate, tolerance, max_iter=1000)`: Implements single-variable linear regression using Gradient Descent.
    - `linear_regression_2d(X1, X2, Y, learning_rate, tolerance, max_iter=1000)`: Implements multiple-variable linear regression using Gradient Descent.

3. **Visualization Functions**:
    - `plot_1d_regression(X, Y, w, b, title)`: Plots the fitted line and data points for single-variable linear regression.
    - `plot_2d_regression(X1, X2, Y, w1, w2, b, title)`: Plots the 3D surface and data points for multiple-variable linear regression.

4. **Main Functions for Task Execution**:
    - `task1()`: Executes single-variable linear regression on a predefined dataset.
    - `task2()`: Executes multiple-variable linear regression on a predefined dataset.

5. **User Menu**:
    - `main()`: Provides a simple menu to select which task to run.

---

## Requirements

The project requires the following Python libraries:

- `numpy`
- `matplotlib`

You can install the required libraries using the following command:

```bash
pip install numpy matplotlib