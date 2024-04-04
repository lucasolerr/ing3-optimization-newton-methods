import numpy as np


def f1(x1, x2):
    return x1**2 + x2**2 - 5


def f2(x1, x2):
    return x1**2 - x2**2 + 3


# Calcul de la matrice jacobienne A
def A(x1, x2):
    return np.array([[2 * x1, 2 * x2], [2 * x1, -2 * x2]])


# Calcul de la matrice B
def B(x1, x2):
    return np.array([-f1(x1, x2), -f2(x1, x2)])


# Calcul de la matrice y
def y(A, B):
    return np.linalg.solve(A, B)


# Algo de Newton en 2D
def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0
    delta_x = np.ones_like(x)
    n = 0
    while n < max_iter and np.linalg.norm(delta_x) > tol:
        delta_x = y(A(x[0], x[1]), B(x[0], x[1]))
        x = x + delta_x
        n += 1
    return x, n


# Valeur de départ
x1 = float(input("x1:"))
x2 = float(input("x2:"))
x0 = np.array([x1, x2])

solution, n = newton_method(x0)
print("Solution:", solution, "en", n, "étapes.")
