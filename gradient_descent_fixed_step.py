import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2 + 5 * np.cos(x)


def df(x):
    return 2 * x - 5 * np.sin(x)


def omega(x):
    return -df(x)


def gradient_descent(x0, alpha, tol=1e-6, max_iter=100):
    x = x0
    n = 0
    residual = 2 * tol
    updates = [(x, f(x))]
    while residual >= tol and n < max_iter:
        x_next = x + alpha * omega(x)
        residual = abs(x_next - x)
        x = x_next
        n += 1
        updates.append((x, f(x)))

    return x, n, updates


# Valeur de départ
x0 = float(input("x0:"))
alpha = 0.1
solution, n, updates = gradient_descent(x0, alpha)
print("Solution:", solution, "en", n, "étapes.")

# Tracer le graphique. On centre les valeurs autour de la solution
x_values = np.linspace(-solution - x0, solution + x0, 400)
y_values = f(x_values)
plt.plot(x_values, y_values, label="f(x)")

x_updates = [update[0] for update in updates]
y_updates = [update[1] for update in updates]
plt.plot(x_updates, y_updates, "or")

plt.title("Algorithme de la descente du gradient à pas fixe avec α=0.1")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
