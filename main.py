import numpy as np

# Define a function (example: f(x) = x^2)
def f(x):
    return x**2

# Generate x values
x = np.linspace(-10, 10, 100)

# Compute f(x)
y = f(x)

# Compute integral of f(x) = x^2
integral = np.trapezoid(y, x)

print("Integral of f(x) = x^2 over [-10, 10]:", integral)
