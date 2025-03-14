import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. NumPy: створення набору даних
size = 100  # можна змінити
X = np.arange(size)
y_true = 2 * X + 5 + np.random.normal(0, 10, size)  # лінійна функція з шумом

# Виконання CRUD операцій
scalar = np.random.randint(1, 100)
vector = np.random.randint(1, 100, size)
matrix = np.random.randint(1, 100, (10, 10))

# 2. Pandas: створення DataFrame

data = pd.DataFrame({'X': X, 'Y': y_true})
print(data.head())
print(data.describe())
print(data.iloc[:5])
print(data.loc[data['Y'] > 50])

# 3. Matplotlib: побудова графіка та обчислення помилок
coefficients = np.polyfit(X, y_true, 1)
y_pred = np.polyval(coefficients, X)

plt.scatter(X, y_true, label="Дані")
plt.plot(X, y_pred, color='red', label="Лінійна регресія")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Обчислення MAE та MSE
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
print(f"MAE: {mae}, MSE: {mse}")

# Запис результатів у CSV
results = pd.DataFrame({'X': X, 'Y': y_true, 'Y_hat': y_pred, 'MAE': mae, 'MSE': mse})
results.to_csv("results.csv", index=False)
