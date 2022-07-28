import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def fact(n):
    return 0 if n < 1 else n * fact(n - 1)

# пункт 1
print('STEP 1:')
data = pd.read_excel('data.xlsx')
X = np.array(data[1])
T = list(range(1, len(X) + 1))

avg = sum(X) / len(X)
var = sum([(x - avg) ** 2 for x in X]) / len(X)

print('M[X]:\t', round(avg, 2))
print('D[X]:\t', round(var, 2))
print()

plt.figure(figsize=(12, 7))
plt.plot(T, X)
plt.plot(T, X, '.')
plt.legend()
plt.grid(True)
plt.show()

# пункт 2
print('STEP 2:')
alpha1 = 0.1
beta1 = 1 - alpha1

alpha2 = 0.3
beta2 = 1 - alpha2

S1 = [np.mean(X[:5])]
S2 = [np.mean(X[:5])]

e1 = [X[0] - S1[0]]
e2 = [X[0] - S2[0]]

table = PrettyTable()
table.field_names = ['t', 'x', f's ({alpha1})', f'e ({alpha1})', f's ({alpha2})', f'e ({alpha2})']
table.add_row([1, X[0], round(S1[0], 2), round(e1[0], 2), round(S2[0], 2), round(e2[0], 2)])

for t in range(1, len(X)):
    S1.append(alpha1 * X[t] + beta1 * S1[t - 1])
    S2.append(alpha2 * X[t] + beta2 * S2[t - 1])

    e1.append(X[t] - S1[-1])
    e2.append(X[t] - S2[-1])

    table.add_row([t + 1, X[t], round(S1[t], 2), round(e1[t], 2), round(S2[t], 2), round(e2[t], 2)])

print(table)

print(f'AVG_E (alpha = {alpha1}): {round(np.mean(e1), 4)}')
print(f'AVG_E (alpha = {alpha2}): {round(np.mean(e2), 4)}')

var_e1 = sum([e ** 2 for e in e1]) / (len(X) - 1)
var_e2 = sum([e ** 2 for e in e2]) / (len(X) - 1)
print(f'S2[^xt](alpha = {alpha1}): {round(var_e1, 4)}')
print(f'S2[^xt](alpha = {alpha2}): {round(var_e2, 4)}')
print()

# пункт 3 (МНК)
print('STEP 3:')
N = len(X)
a1 = (N * sum([X[i] * T[i] for i in range(N)]) - sum(X) * sum(T)) / (N * sum([t ** 2 for t in T]) - sum(T) ** 2)
a0 = (sum(X) - a1 * sum(T)) / N

str_sign = '+' if a1 >= 0 else '-'
print(f'x = {round(a0, 2)} {str_sign} {round(a1, 2)} * t')
print()

Y = [(a0 + a1 * t) for t in T]

plt.figure(figsize=(12, 7))
plt.plot(T, X, '.')
plt.plot(T, X, label='Факт.')
plt.plot(T, Y, '.')
plt.plot(T, Y, label='МНК')
plt.legend()
plt.grid(True)
plt.show()

# пункт 4 (эксп. сглаживание)
print('STEP 4:')
S11 = [a0 - (beta1 / alpha1) * a1]
S12 = [a0 - (2 * beta1 / alpha1) * a1]
S21 = [a0 - (beta2 / alpha2) * a1]
S22 = [a0 - (2 * beta2 / alpha2) * a1]

for t in range(1, len(X)):
    S11.append(alpha1 * X[t] + beta1 * S11[t - 1])
    S12.append(alpha1 * S11[t] + beta1 * S12[t - 1])

    S21.append(alpha2 * X[t] + beta2 * S21[t - 1])
    S22.append(alpha2 * S21[t] + beta2 * S22[t - 1])

plt.figure(figsize=(12, 7))
plt.plot(T, X, '.')
plt.plot(T, X, label='Факт.')
plt.plot(T, S11, label=f'n = 1, alpha = {alpha1}')
plt.plot(T, S12, label=f'n = 1, alpha = {alpha2}')
plt.plot(T, S21, label=f'n = 2, alpha = {alpha1}')
plt.plot(T, S22, label=f'n = 2, alpha = {alpha2}')
plt.legend()
plt.grid(True)
plt.show()

print()

# пункт 5 (линейная модель, прогноз при m = 1)
print('STEP 5:')
A01 = [2 * S11[0] - S12[0]]
A11 = [(S11[0] - S12[0]) * alpha1 / beta1]

A02 = [2 * S21[0] - S22[0]]
A12 = [(S21[0] - S22[0]) * alpha2 / beta2]

m = 1

e1 = [0 for _ in range(m)]
e2 = [0 for _ in range(m)]

table = PrettyTable()
table.field_names = ['t', 'x', f'predict x ({alpha1})', f'e ({alpha1})', f'predict x ({alpha2})', f'e ({alpha2})']

for i in range(m):
    table.add_row([i + 1, X[i], np.NaN, np.NaN, np.NaN, np.NaN])

for t in range(m, len(X)):
    A01.append(2 * S11[t] - S12[t])
    A11.append((S11[t] - S12[t]) * alpha1 / beta1)

    A02.append(2 * S21[t] - S22[t])
    A12.append((S21[t] - S22[t]) * alpha2 / beta2)

    _x1 = A01[t - m] + A11[t - m] * m
    _x2 = A02[t - m] + A12[t - m] * m

    e1.append(X[t] - _x1)
    e2.append(X[t] - _x2)

    table.add_row([t + 1, X[t], round(_x1, 2), round(X[t] - _x1, 2), round(_x2, 2), round(X[t] - _x2, 2)])

print(table)

print(f'AVG_E (alpha = {alpha1}): {round(np.mean(e1[m:]), 4)}')
print(f'AVG_E (alpha = {alpha2}): {round(np.mean(e2[m:]), 4)}')

var_e1 = sum([e ** 2 for e in e1[m:]]) / (len(e1[m:]) - 2)
var_e2 = sum([e ** 2 for e in e2[m:]]) / (len(e2[m:]) - 2)
print(f'S2[^xt](alpha = {alpha1}): {round(var_e1, 4)}')
print(f'S2[^xt](alpha = {alpha2}): {round(var_e2, 4)}')
print()

# пункт 6 (линейная модель, прогноз при m = 5)
print('STEP 6:')
m = 5

e1 = [0 for _ in range(m)]
e2 = [0 for _ in range(m)]

table = PrettyTable()
table.field_names = ['t', 'x', f'predict x ({alpha1})', f'e ({alpha1})', f'predict x ({alpha2})', f'e ({alpha2})']

for i in range(m):
    table.add_row([i + 1, X[i], np.NaN, np.NaN, np.NaN, np.NaN])

for t in range(m, len(X)):
    A01.append(2 * S11[t] - S12[t])
    A11.append((S11[t] - S12[t]) * alpha1 / beta1)

    A02.append(2 * S21[t] - S22[t])
    A12.append((S21[t] - S22[t]) * alpha2 / beta2)

    _x1 = A01[t - m] + A11[t - m] * m
    _x2 = A02[t - m] + A12[t - m] * m

    e1.append(X[t] - _x1)
    e2.append(X[t] - _x2)

    table.add_row([t + 1, X[t], round(_x1, 2), round(X[t] - _x1, 2), round(_x2, 2), round(X[t] - _x2, 2)])

print(table)

print(f'AVG_E (alpha = {alpha1}): {round(np.mean(e1[m:]), 4)}')
print(f'AVG_E (alpha = {alpha2}): {round(np.mean(e2[m:]), 4)}')

var_e1 = sum([e ** 2 for e in e1[m:]]) / (len(e1[m:]) - 2)
var_e2 = sum([e ** 2 for e in e2[m:]]) / (len(e2[m:]) - 2)
print(f'S2[^xt](alpha = {alpha1}): {round(var_e1, 4)}')
print(f'S2[^xt](alpha = {alpha2}): {round(var_e2, 4)}')
print()

# пункт 7 (выбор модели и порядка полинома)
print('STEP 7:')
print('N\tdelta\n')
dx = [(X[t] - X[t - 1]) for t in range(1, len(X))]
dxs = [(np.mean(dx), dx)]
print(f'0\t{dxs[-1][0]}')
for i in range(7):
    x = dxs[-1][1]
    dx = [(x[t] - x[t - 1]) for t in range(1, len(x))]
    dxs.append((np.mean(dx), dx))
    print(f'{i + 1}\t{dxs[-1][0]}')