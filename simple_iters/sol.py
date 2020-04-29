#!/usr/bin/env python
import numpy as np
from sympy import (
    symbols,
    Eq,
    solve
)
from colorama import (
    Fore,
    Style
)


def logg(*args, **kwargs):
    print(f'\n[ {Fore.BLUE}{Style.BRIGHT}LOG{Style.RESET_ALL} ] ', end='')
    print(*args, **kwargs)


def mkiter(last):
    return np.array([
        round(
            float(sol[i].subs(
                [(x[j], last[j-1]) for j in hrange(3)]
            )),
            5
        ) for i in hrange(3)
    ])


def hrange(n):
    """ Human range """
    return range(1, n+1)


# eps
eps = .01
# symbols x1, x2, x3
x = {
    i: symbols(f'x{i}') for i in hrange(3)
}
# expessions
expr = {
    1: 175.1*x[1] + 28*x[2] + 47*x[3],
    2: x[1] + 18.5*x[2] + 8.01*x[3],
    3: 6.1*x[1] + 12.002*x[2] + 48*x[3]
}
# equalities
eq = {
    1: Eq(expr[1], 269.1),
    2: Eq(expr[2], 17.02),
    3: Eq(expr[3], 102.1)
}
# solutions of eq[i] for x[i]
sol = {
    i: solve(eq[i], x[i])[0] for i in hrange(3)
}

print(solve(
    (eq[1], eq[2], eq[3]),
    (x[1], x[2], x[3])
))

# extracting coeffs
x_coeffs = []
frees = []
for i in sol:
    tmp_expr = sol[i]
    coeffs = tmp_expr.as_coefficients_dict()
    x_coeffs.append(
        [float(coeffs.get(x[i], 0)) for i in x]
    )
    frees.append(float(coeffs.get(1, 0)))

# matrix of x coeffs
A = np.around(np.array(x_coeffs), 5)
# matrix of free coeffs
B = np.around(np.array(frees), 5)

logg('A:')
print(A)
logg('B:')
print(B)

# A norm
A_norm = np.linalg.norm(A, 1)
logg('||A||:', A_norm)

if A_norm >= 1: # needed condition
    exit()

# x matrix
xmx = [B]

xmx.append(mkiter(xmx[-1]))

logg('xmx[0]:')
print(xmx[0])

logg('xmx[1]:')
print(xmx[1])

diff = np.around(xmx[1] - xmx[0], 5)
logg('xmx[1] - xmx[0]')
print(diff)

diff_norm = np.linalg.norm(diff, 1)
logg('||xmx[1] - xmx[0]||:', diff_norm)

m = np.log((eps * (1 - A_norm)) / diff_norm) / np.log(A_norm)
logg('m:', m)

for i in range(2, np.int_(m) + 2):
    print('\n\n\n')
    logg('iteration number', i)
    xmx.append(mkiter(xmx[-1]))

    logg(f'xmx[{i-1}]:')
    print(xmx[i-1])

    logg(f'xmx[{i}]:')
    print(xmx[i])

    diff = np.around(xmx[i] - xmx[i-1], 5)
    logg(f'xmx[{i}] - xmx[{i-1}]')
    print(diff)

    diff_norm = np.linalg.norm(diff, 1)
    logg(f'||xmx[{i}] - xmx[{i-1}]||:', diff_norm)

