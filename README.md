# DSML_Assignment1
First Assignment in DSML Courses


Here’s an overview of the exercise, the implementation, and the thought process behind how we approached each task, which you can add as a file to your project:

---

## Gradient Descent Exercise - Explanation and Implementation

### Overview
The purpose of this exercise is to implement and explore gradient descent in 1D to minimize a quadratic function, analyze the effect of learning rates, starting points, and perform stability analysis around the minimum.

### Function:
The function given is:

\[
f(x) = x^2 + 3x + 2
\]

It's a simple quadratic function with a minimum that can be found analytically as well as using gradient descent. The goal is to perform gradient descent to minimize this function and investigate how the parameters like learning rate and initial points affect the behavior of the algorithm.

---

### **Task 1: Gradient Descent for Minimization**

#### Goal:
Perform gradient descent with a learning rate of \( \eta = 0.1 \) starting from \( x_0 = 10 \) and iterating until the gradient's absolute value is less than \( 10^{-6} \).

#### Thought Process:
1. We defined the function \( f(x) \) and its derivative \( f'(x) = 2x + 3 \), which is needed for gradient descent.
2. The gradient descent loop was set to update \( x \) iteratively
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
