#!/usr/bin/env python
from sympy import symbols, Eq, solve
import numpy as np
from numpy.linalg import norm


def build_eq(coeffs, free):
    return Eq(
        sum([coeffs[i] * x[i+1] for i in range(4)]),
        free
    )


def prep(v):
    return round(float(v), 5)


def build_AB():
    A = np.array([])
    B = np.array([])

    for i in sol:
        coeffs = sol[i].as_expr().as_coefficients_dict()
        x_coeffs = [prep(coeffs.get(x[i], 0)) for i in x]

        A = np.append(A, x_coeffs)
        B = np.append(B, prep(coeffs.get(1, 0)))

    A = A.reshape((4, 4))

    return A, B


def mkiter():
    return np.array([
        prep(
            sol[i].subs(
                [(x[j], xmx[-1][j-1]) for j in range(1, 5)]
            )
        ) for i in range(1, 5)
    ])


def intro():
    print('Матрица альфа:')
    print(A, end='\n\n')

    print('Матрица бета:')
    print(B, end='\n\n')

    q = norm(A, 1)
    print('Норма матрицы альфа:')
    print(q, end='\n\n')

    if q >= 1:
        print('Норма больше единицы, можно не решать')
        exit()

    # matrix of x -s
    xmx.append(mkiter())

    print('x(0) это матрица бета, x(1) равно:')
    print(xmx[1], end='\n\n')

    diff = xmx[1] - xmx[0]
    print('Разность x(1) - x(0):')
    print(diff, end='\n\n')

    diff_norm = prep(norm(diff, 1))
    print('Норма разности ||x(1) - x(0)||:')
    print(diff_norm, end='\n\n')


def mkiters():
    i = 0
    while prep(norm(xmx[-1] - xmx[-2], 1)) > eps:
        i += 1

        print('Итерация', i)
        print('===========')

        xmx.append(mkiter())
        print(f'x({i}):')
        print(xmx[-1], end='\n\n')

        diff_norm = prep(norm(xmx[-1] - xmx[-2], 1))
        print(f'||x({i}) - x({i-1})||:')
        print(diff_norm, end='\n\n')


def main():
    intro()

    input('Нажмите Enter чтоб начать итерирование')
    print('='*21)
    print(' НАЧАЛО ИТЕРИРОВАНИЯ ')
    print('='*21, end='\n\n\n')

    mkiters()

    print('Разность нормы меньше eps, конец итерирования')

    print('=============')
    print('=== ОТВЕТ ===')
    print('=============', end='\n\n')

    print('\n'.join([f'x{i+1} = {xmx[-1][i]}' for i in range(4)]))


eps = .01

x = {
    i: symbols(f'x{i}') for i in range(1, 5)
}

eq = {
    1: build_eq((-.77, -.04, .21, -.18), -1.24),
    2: build_eq((.25, -1.23, .16, -.09), 1.12),
    3: build_eq((-.21, .16, .8, -.13), 2.56),
    4: build_eq((.15, -.31, .06, 1.12), -.77),
}

sol = {
    i: solve(eq[i], x[i])[0] for i in range(1, 5)
}

A, B = build_AB()

xmx = [B]


if __name__ == '__main__':
    main()

