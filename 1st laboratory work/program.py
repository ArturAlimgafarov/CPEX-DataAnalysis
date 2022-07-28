import pandas as pd
import numpy as np
from math import sqrt

# загружаем данные
data_frame = pd.read_excel('test.xlsx')

# а) Вывод средних и дисперсий по столбцам
col_names = data_frame.columns.ravel()[1:]
col_avg = []
col_s2 = []
for attr in col_names:
    col_values = data_frame[attr]
    col_avg.append(np.average(col_values))
    col_s2.append(np.average([(z - col_avg[-1]) ** 2 for z in col_values]))
    # print(f'{attr}: {col_avg[-1]}, {col_s[-1]}')

pd.DataFrame({
    'Среднее': col_avg,
    'Дисперсия': col_s2,
}, index=col_names).to_excel('Средние и дисперсии по столбцам.xlsx')


# б) Вывод стандартизированной матрицы
n = len(data_frame[col_names[0]])
p = len(col_names)
X = np.zeros((n, p))
for i in range(n):
    for j in range(p):
        X[i, j] = (data_frame[col_names[j]][i] - col_avg[j]) / sqrt(col_s2[j])

pd.DataFrame(X).to_excel('Стандартизированная матрица.xlsx')


# в) Вывод ковариационной матрицы
E = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        e = 0
        for k in range(n):
            e += (data_frame[col_names[i]][k] - col_avg[i]) * (data_frame[col_names[j]][k] - col_avg[j])
        E[i, j] = e / n

pd.DataFrame(E).to_excel('Ковариационная матрица.xlsx')


# г) Вывод корреляционной матрицы
R = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        r = 0
        for k in range(n):
            r += X[k, i] * X[k, j]
        R[i, j] = r / n

pd.DataFrame(R).to_excel('Корреляционная матрица.xlsx')


# д) Анализ
alpha = 0.05
# table = pd.read_excel('t-критерий.xlsx')
# t_tabl = table[alpha][p - 2]
t_tabl = 1.9944371
print(f'Табличное значение t-критерия Стьюдента ({alpha}, {n - 2}): {t_tabl}')
t_calc = np.zeros((p, p))
for i in range(p):
    mass = []
    for j in range(p):
        r = R[i, j]
        t_calc[i, j] = np.NAN if i == j else r * sqrt(n - 2) / sqrt(1 - r ** 2)
        if i == j:
            mass.append('NA')
        else:
            if abs(t_calc[i, j]) >= t_tabl:
                mass.append(1)
            else:
                mass.append(0)
    print(mass)

pd.DataFrame(t_calc).to_excel('Матрица статистик.xlsx')













# hyps = []
# hyps.append(['NA', 1, 0, 1, 0, 0, 0, 0, 0, 0])
# hyps.append([1, 'NA', 0, 1, 0, 1, 0, 0, 1, 1])
# hyps.append([0, 0, 'NA', 0, 1, 1, 0, 0, 0, 1])
# hyps.append([1, 1, 0, 'NA', 1, 0, 1, 1, 1, 0])
# hyps.append([0, 0, 1, 1, 'NA', 0, 1, 1, 1, 1])
# hyps.append([0, 1, 1, 0, 0, 'NA', 1, 1, 0, 0])
# hyps.append([0, 0, 0, 1, 1, 1, 'NA', 0, 0, 0])
# hyps.append([0, 0, 0, 1, 1, 1, 0, 'NA', 0, 0])
# hyps.append([0, 1, 0, 1, 1, 0, 0, 0, 'NA', 0])
# hyps.append([0, 1, 1, 0, 1, 0, 0, 0, 0, 'NA'])
#
# for h in hyps:
#     print(h)
