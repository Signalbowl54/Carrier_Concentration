import numpy as np

def trap_integral(func, a: float, b: float, n: int, *args):
    """
    Calculates the numerical integral using the trapezoidal rule
    :param func: Takes in a function to be integrated
    :param a: Starting point
    :param b: Ending point
    :param n: number of intervals
    :param args: any additional variable values
    :return: Area approximation
    """
    h = (b-a) / n
    area = .5 * (func(b, *args) - func(a, *args))
    for i in range(1, n):
        x_i = a + i * h
        area += func(x_i, *args)

    return area * h


def trap_vector(func, a: float, b: float, n: int, *args):
    """
    Uses a vectorized approach to calculate the numerical integral using the trapezoidal rule
    :param func: Takes in a function to be integrated
    :param a: Starting point
    :param b: Ending point
    :param n: number of intervals
    :param args: any additional variable values
    :return: Area approximation
    """
    h = (b-a) / n
    x=np.linspace(a,b,n+1)

    y= func(x, *args)

    area = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

    return area

def simpson_integral(func, a: float, b: float, n: int, *args):
    """
    Calculates the numerical integral using Simpson's rule
    :param func: Takes in a function to be integrated
    :param a: Starting point
    :param b: Ending point
    :param n: number of intervals
    :param args: any additional variable values
    :return: Area approximation
    """
    if n % 2 !=0:
        raise ValueError("n must be even")

    h = (b-a) / n
    x = np.linspace(a,b,n+1)

    area = (func(a, *args) + func(b, *args))
    for i in range(1, n):
        if i % 2 == 0:
            area += 2*func(x[i], *args)
        else:
            area += 4*func(x[i], *args)

    return area * (h/3)

def simpson_vector(func, a: float, b: float, n: int, *args):
    """
    Uses a vectorized approach to calculate the numerical integral using Simpson's rule
    :param func: Takes in a function to be integrated
    :param a: Starting point
    :param b: Ending point
    :param n: number of intervals
    :param args: any additional variable values
    :return: Area approximation
    """
    if n % 2 !=0:
        raise ValueError("n must be even")

    h = (b-a) / n
    x = np.linspace(a,b,n+1)

    y = func(x, *args)

    area = (h/3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

    return area

def central_difference(func, x: float, *args, dx=1.602e-24):
    """
    Calculates the central difference approximation for numerical differentiation
    :param func: Takes in a function to be differentiated
    :param x: The point at which to calculate the central difference
    :param dx: The change in x (defaults to 1e-5)
    :param args: any additional variable values
    :return: Numerical derivative approximation
    """
    f_forwards = func(x + dx, *args)

    f_backwards = func(x - dx, *args)

    return (f_forwards - f_backwards)/(2*dx)


def newton_raphson(func, x: float, *args, tol=1.602e-25, max_iter=1000):
    """
    Uses the Newton Raphson Method to find the roots of a function given an initial approximation
    :param func: Takes in a function to approximate roots
    :param x: approximation point
    :param args: any additional variable values
    :param tol: Tolerance
    :param max_iter: maximum number of iterations
    :return: approximated root
    """

    for i in range(max_iter):
        f_val = func(x, *args)
        f_prime = central_difference(func, x, *args)

        if abs(f_prime) < 1e-30:
            print("Numerical derivative it too small. Divergence likely.")
            return None
        h = f_val / f_prime
        x = x - h

        if abs(h) < tol:
            return x
    print(f"Failed to converge after {max_iter} iterations.")
    return x


def bisection_method(func, lower_bound: float, upper_bound: float, *args, tol=1.602e-25):
    """
    Uses the bisection method to find the roots of a function given an upper and lower bound
    :param func: Takes in a function to approximate roots
    :param lower_bound: lower bound of approximation
    :param upper_bound: upper bound of approximation
    :param args: any additional variable values
    :param tol: Tolerance
    :return: approximated root
    """
    a = lower_bound
    b = upper_bound

    if  (func(a, *args) * func(b, *args)) > 0:
        print("Error: Root is not bracketed between the bounds.")
        return None
    i=0
    while (b-a)/2 > tol:
        c = (a + b) / 2.0
        f_c = func(c, *args)

        if f_c == 0:
           #print(f'Final result:{c}')
           return c

        if (func(a, *args) * func(c, *args)) < 0:
            b = c
            #print(f'upper bound:{b}\niter:{i}')
        else:
            a = c
            #print(f'lower bound:{a}\niter:{i}')
        i += 1

    return (a + b) / 2.0






























