import numpy as np
import pandas as pd
import math

test_matrix = np.array([
    [1, 0.42, 0.54, 0.66],
    [0.42, 1, 0.32, 0.44],
    [0.54, 0.32, 1, 0.22],
    [0.66, 0.44, 0.22, 1]
])

# 1. импортирование данных
data = pd.read_excel('test.xlsx')
col_names = data.columns.ravel()[1:]
X = []
for attr in col_names:
    X.append(data[attr])

# 2. стандартизация
col_avg = []
col_s2 = []
for attr in col_names:
    col_values = data[attr]
    col_avg.append(np.average(col_values))
    col_s2.append(np.average([(z - col_avg[-1]) ** 2 for z in col_values]))

n = len(data[col_names[0]])
p = len(col_names)
X = np.zeros((n, p))
for i in range(n):
    for j in range(p):
        X[i, j] = (data[col_names[j]][i] - col_avg[j]) / math.sqrt(col_s2[j])

# 3. ковариационная матрица
R = np.cov(X.transpose())

def jacobi(A, eps=1e-3):
    A1 = A.copy()
    N = len(A)

    # step 1
    T = np.eye(N)

    def sign(x):
        return 1 if x >= 0 else -1
    def check():
        for i in range(N):
            for j in range(N):
                if (i != j) and (abs(A1[i, j]) >= eps):
                    return False
        return True
    def transform(p, q):
        y = (A1[p, p] - A1[q, q]) / 2
        x = -1 if y == 0 else sign(y) * A1[p, q] / math.sqrt(A1[p, q] ** 2 + y ** 2)

        s = x / math.sqrt(2 * (1 + math.sqrt(1 - x ** 2)))
        c = math.sqrt(1 - s ** 2)

        for i in range(N):
            if (i != p) and (i != q):
                z1, z2 = A1[i, p], A1[i, q]
                A1[q, i] = z1 * s + z2 * c
                A1[i, q] = A1[q, i]
                A1[i, p] = z1 * c - z2 * s
                A1[p, i] = A1[i, p]

        z5 = s ** 2
        z6 = c ** 2
        z7 = s * c
        v1 = A1[p, p]
        v2 = A1[p, q]
        v3 = A1[q, q]

        A1[p, p] = v1 * z6 + v3 * z5 - 2 * v2 * z7
        A1[q, q] = v1 * z5 + v3 * z6 + 2 * v2 * z7
        A1[p, q] = (v1 - v3) * z7 + v2 * (z6 - z5)
        A1[q, p] = A1[p, q]

        for i in range(N):
            z3, z4 = T[i, p], T[i, q]
            T[i, q] = z3 * s + z4 * c
            T[i, p] = z3 * c - z4 * s

    # step 2
    a0 = 0
    for i in range(N):
        for j in range(N):
            a0 += A1[i, j] ** 2
    a0 = math.sqrt(a0) / N

    ak = a0
    while True:
        # 3 step
        p = -1
        q = -1
        a_max = 0
        for i in range(N):
            for j in range(N):
                if (i != j) and (abs(A1[i, j]) > abs(a_max)):
                    a_max = A1[i, j]
                    p, q = i, j

        # step 4
        if (abs(a_max) > ak):
            transform(p, q)
            ak /= n ** 2

        # step 5
        if check():
            break

    d = 5
    eigens = [(round(A1[i, i], d), [round(t, d) for t in T[:, i]]) for i in range(N)]
    eigens.sort(key=lambda item: item[0], reverse=True)

    return eigens

# 4. вычисление статистики
N = len(R)
chi_sqr = 16.919
d = 0
for i in range(N):
    for j in range(N):
        if i != j:
            d += R[i, j] ** 2
d = round(d * N, 5)
print('Статистика d =', d)
print('Хи2 (0.05) =', chi_sqr)
print()

# v, m = np.linalg.eig(R)
#
# for i in range(len(R)):
#     print(f'{round(v[i], 5)}: {[round(e, 5) for e in m[:, i]]}')
# print()

lmbds = []
eig_vects = []
for item in jacobi(R):
    lmbds.append(item[0])
    eig_vects.append(item[1])
    print(f'{item[0]}: {item[1]}')
print()

p = 0
s = sum(lmbds)
for i in range(1,N):
    if sum(lmbds[:i]) / s > 0.95:
        p = i
        break
print('Число новых признаков (главных компонент): ', p)
print()

for i in range(p):
    str_pc = f'pc{i + 1} = '
    str_pc += (f'-{abs(eig_vects[i][0])}x{1} ' if eig_vects[i][0] < 0 else f'{eig_vects[i][0]}x{1} ')
    for j in range(1, N):
        str_pc += (f'- {abs(eig_vects[i][j])}x{j + 1} ' if eig_vects[i][j] < 0 else f'+ {eig_vects[i][j]}x{j + 1} ')
    print(str_pc)