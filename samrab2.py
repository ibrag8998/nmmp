#!/usr/bin/env python
from sympy import symbols, Eq, solve
import numpy as np
from numpy.linalg import norm


def ask_n():
    print('Введите N (номер варианта).'
          ' Или просто нажмите Enter, чтобы использовать'
          ' вариант 8 (Гаджимагомедов Ибрагим)')
    n = input('N = ')
    try:
        n = int(n)
    except:
        print('Используем вариант по умолчанию: 8')
        n = 8

    return n


def build_eq(coeffs, free):
    return Eq(
        sum([coeffs[i] * x[i+1] for i in range(N)]),
        free
    )


def prep(n):
    return round(float(n), 5)


def build_AB():
    A = np.array([])
    B = np.array([])

    for i in sol:
        coeffs = sol[i].as_expr().as_coefficients_dict()
        x_coeffs = [prep(coeffs.get(x[i], 0)) for i in x]

        A = np.append(A, x_coeffs)
        B = np.append(B, prep(coeffs.get(1, 0)))

    A = A.reshape((N, N))

    return A, B


def mkiter_zdl():
    new_x = [0] * N
    xmx_stream = xmx[-1].tolist().copy()
    for i in range(N):
        new_x[i] = prep(sol[i+1].subs([(x[j], (
            xmx_stream[j-1] if j-1 < i \
            else xmx[-1][j-1]
        )) for j in range(1, N+1)]))

        xmx_stream.append(new_x[i])
    return np.array(new_x)


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
    xmx.append(mkiter_zdl())

    print('x(0) это матрица бета, x(1) равно:')
    print(xmx[1], end='\n\n')

    diff = xmx[1] - xmx[0]
    print('Разность x(1) - x(0):')
    print(diff, end='\n\n')

    diff_norm = prep(norm(diff, 1))
    print('Норма разности ||x(1) - x(0)||:')
    print(diff_norm, end='\n\n')


def mkiters():
    i = 1
    while prep(norm(xmx[-1] - xmx[-2], 1)) > eps:
        i += 1
        print('Итерация', i)
        print('===========')

        xmx.append(mkiter_zdl())

        print(f'x({i}):')
        print(xmx[-1], end='\n\n')

        diff_norm = prep(norm(xmx[-1] - xmx[-2], 1))
        print(f'||x({i}) - x({i-1})||:')
        print(diff_norm, end='\n\n')


def main(method='1'):
    intro()

    input('Нажмите Enter для начала итерирования')
    print('='*21)
    print(' НАЧАЛО ИТЕРИРОВАНИЯ ')
    print('='*21, end='\n\n\n')

    mkiters()

    print('Норма разности меньше eps, конец итерирования', end='\n\n')
    print('=============')
    print('=== ОТВЕТ ===')
    print('=============', end='\n\n')

    print('\n'.join([f'x{i+1} = {xmx[-1][i]}' for i in range(N)]))

eps = .01

N = 3

x = {i: symbols(f'x{i}') for i in range(1, N+1)}

n = ask_n()

coeffs = [
    [20*n + 15.1, 3*n + 4, 5*n + 7],
    [1, 10.5 + n, n + .01],
    [n - 1.9, n + 4.002, 16 + 4*n]
]
free = [30*n + 29.1, 2*n + 1.02, 9*n + 30.1]

eq = {i+1: build_eq(coeffs[i], free[i]) \
        for i in range(N)}


sol = {i: solve(eq[i], x[i])[0] for i in range(1, N+1)}

A, B = build_AB()

xmx = [B]

main()

