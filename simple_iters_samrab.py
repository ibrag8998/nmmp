#!/usr/bin/env python
from sympy import symbols, Eq, solve
import numpy as np
from numpy.linalg import norm


def ask_n():
    print('На данный момент поддерживаются'
          ' только такие системы, в которых'
          ' в которых кол-во уравнений равно'
          ' кол-ву неизвестных')
    print('Сколько уравнений в системе (2-10)?')

    n = int(input())
    while n < 2 or n > 10:
        n = int(input('От 2 до 10: '))

    return n


def ask_coeffs():
    coeffs = []
    for i in range(1, N+1):
        print('Введите через пробел коэффициенты'
              ' уравнения номер', i)
        coeffs.append(list(map(
            prep,
            input().split()
        )))
        assert len(coeffs[-1]) == N, 'Неверное кол-во коэф-ов'

    return coeffs


def ask_free():
    print('Введите через пробел свободные коэф-ты')
    free = list(map(prep, input().split()))
    assert len(free) == N, 'Неверное кол-во коэф-ов'
    return free


def build_eq(coeffs, free):
    return Eq(
        sum([coeffs[i] * x[i+1] for i in range(4)]),
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

    A = A.reshape((4, 4))

    return A, B


def mkiter():
    return np.array([
        prep(
            sol[i].subs(
                [(x[j], xmx[-1][j-1]) for j in range(1, 5)]
            )) for i in range(1, 5)])


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
    i = 2
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

    input('Нажмите Enter для начала итерирования')
    print('='*21)
    print(' НАЧАЛО ИТЕРИРОВАНИЯ ')
    print('='*21, end='\n\n\n')

    mkiters()

    print('Норма разности меньше eps, конец итерирования', end='\n\n')
    print('=============')
    print('=== ОТВЕТ ===')
    print('=============', end='\n\n')

    print('\n'.join([f'x{i+1} = {xmx[-1][i]}' for i in range(4)]))


eps = .01


#N = ask_n()
N = 4

x = {i: symbols(f'x{i}') for i in range(1, N+1)}

#coeffs = ask_coeffs()
#free = ask_free()


for m, n, p in [
    [-.77, .16, 1.12],
    [.93, .07, -.84],
    [-1.14, -.17, .95],
    [1.08, .22, -1.16],
    [.87, -.19, 1.08]
]:
    input('\n\nНажмите Enter для продолжения')
    print('=========')
    print(f'M = {m}\nN = {n}\nP = {p}')
    print('=========')
    input('Нажмите Enter для продолжения')
    coeffs = [
        [m, -.04, .21, -.18],
        [.25, -1.23, n, -.09],
        [-.21, n, .8, -.13],
        [.15, -.31, .06, p]
    ]
    free = [-1.24, p, 2.56, m]

    eq = {i+1: build_eq(coeffs[i], free[i]) \
            for i in range(N)}


    sol = {i: solve(eq[i], x[i])[0] for i in range(1, 5)}

    A, B = build_AB()

    xmx = [B]

    main()

