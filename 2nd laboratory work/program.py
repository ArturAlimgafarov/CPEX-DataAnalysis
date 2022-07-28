import numpy as np
import pandas as pd

data = pd.read_excel('test2.xlsx')
col_names = data.columns.ravel()[1:]

Y = data[col_names[0]]
X = []
for attr in col_names[1:]:
    X.append(data[attr])

Xcopy = X.copy()
X.append([1 for _ in range(len(X[0]))])
Xt = np.array(X)
X = Xt.transpose()
Y = np.array([Y])
a = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, Y.transpose()))
a = [round(float(coef), 3) for coef in a]

c = f'- {abs(a[-1])} ' if a[-1] < 0 else f' {a[-1]} '
eqtn = f'Y = {c}'
print(f'Y: "{col_names[0]}"')
for i in range(1, len(a)):
    print(f'x{i}: "{col_names[i]}"')
    eqtn += (f'- {abs(a[i - 1])}x{i} ' if a[i - 1] < 0 else f'+ {a[i - 1]}x{i} ')

print(f'\nУравнение регрессии: {eqtn}')

y_calc = np.dot(np.array(Xcopy).transpose(), a[:-1]) + a[-1]

err = Y - y_calc

print(f'M(e) = {round(np.mean(err), 4)}')
avgY = np.mean(Y)
R2 = 1 - np.sum(err ** 2) / (np.sum((Y - avgY) ** 2))
print(f'Коэффициент детерминации (R_squared) = {round(R2, 4)}')