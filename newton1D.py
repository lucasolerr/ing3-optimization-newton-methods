def f(x):
    return x**2


def df(x):
    h = 0.00001
    return (f(x + h) - f(x)) / h


def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0
    iter_count = 0
    while iter_count < max_iter:
        fx = f(x)
        dfx = df(x)
        x_next = x - fx / dfx
        if abs(x_next - x) < tol:
            return x, iter_count
        x = x_next
        iter_count += 1


# Valeur de départ
x0 = int(input("Valeur initiale:"))
root, iter_count = newton_method(x0)
print("Racine trouvée en x:", root, "en", iter_count, "étapes.")
