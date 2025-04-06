from sklearn.base import clone
from sklearn.model_selection import cross_validate
from scipy.optimize import minimize
import numpy as np

class GaussianProcess:
    def __init__(self, sigma=1, r=1, noise=0.1):
        self.sigma = sigma # Предполагаемое распределение
        self.r = r # Данный параметр масштабирует значения ковариационной функции
        self.noise = noise # Ввожу шум для борьбы с вырожденными ковариационными матрицами 

    def rbf_kernel(self, xi, xj, sigma=1.0, r=1.0):
        return sigma**2 * np.exp( -np.sum( (xi - xj)**2) / (2 * r**2) ) # Абсолютно гладкое гауссовское ядро
    
    def cov(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return np.array(
            [self.rbf_kernel(x1, x2, self.sigma, self.r) for x1 in X1 for x2 in X2]
        ).reshape( (len(X1), len(X2)) )
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train) 
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        K = self.cov(self.X_train) # Считаем ковариационные матрицы трейна с самим собой
        K_ss = self.cov(X_test) # Считаем ковариационные матрицы теста с самим собой
        K_s = self.cov(self.X_train, X_test) # Считаем ковариационные матрицы трейна и тестом
        K_inv = np.linalg.inv(K + self.noise**2 * np.eye(len(self.X_train))) # Обратная матрица с регуляризационным членом
        mu_s = K_s.T @ K_inv @ self.y_train # Предсказываем среднее
        cov_s = K_ss - K_s.T @ K_inv @ K_s # Предсказываем дисперсии
        return (mu_s, np.diag(cov_s))

class AcquisitionFunc:
    def UCB(self, model_gp, X, b = 2):
        m, var = model_gp.predict(X.reshape(1, -1))
        return -1*(m + b * var)

class BayesOptimization:
    def __init__(self, model,
                       dict_params,
                       scoring,
                       obj_func="GP",
                       acq_func="UCB",
                       n_iters=50,
                       cv=5):
        
        self.n_iters = n_iters
        self.cv = cv
        self.scoring = scoring
        self.obj_func = obj_func
        self.acq_func = acq_func  
        self.model = model
        self.dict_params = dict_params

    def fit(self, X, y):
        number_of_starting_points = 10
        X_sample, Y_sample = [], []
        for point in range(number_of_starting_points): 
            # Задаём каким образом генерирутся оптимизируемые параметры для модели
            uniform_func = np.random.uniform
            # Здесь хранятся случайные значения для оптимизируемых параметров
            optimization_params = {}
            nums_for_params = []
            for key, param in self.dict_params.items(): 
                # Тип переменной
                type_func = param[0]
                # Интервал
                a, b = param[1]
                # Генерация значения гиперпараметра
                num = type_func(uniform_func(a, b))
                optimization_params[key] = num
                nums_for_params.append(num)
            # Кросс-валидация с сгенерированными параметрами 
            model = clone(self.model).set_params(**optimization_params)
            cv_results = cross_validate(model.set_params(**optimization_params),
                                        X, y)
            # Запоминаем новую стартовую точку
            X_sample.append(nums_for_params)
            Y_sample.append(np.mean(cv_results["test_score"]))
        # Перевожу в np массив
        X_sample = np.array(X_sample)
        Y_sample = np.array(Y_sample)
        # Обучение гауссовского процесса
        gp = GaussianProcess(sigma=1, r=1, noise=0.1)
        gp.fit(X_sample, Y_sample)
        # Основной цикл байесовской оптимизации
        for i in range(self.n_iters):
            # Беру максимальный известный y (вообще можно взять любой, но вдруг попаду в максимум)
            y_max = np.max(Y_sample)
            # Генерирую начальные точки для минимизации
            X0 = []
            bounds = []
            for key, param in self.dict_params.items(): 
                # Тип параметра
                type_func = param[0]
                # Интервал
                a, b = param[1]
                # Генерация значения гиперпараметра
                X0.append(type_func(uniform_func(a, b)))
                bounds.append((a, b))
            # Максимизирую функцию приобретения
            result = minimize(AcquisitionFunc.UCB, 
                              x0=X0, 
                              args=(gp, y_max), 
                              bounds=bounds, 
                              method='L-BFGS-B')
            # Следующие X
            X_next = result.x
            # Здесь хранятся значения новой точки для оптимизируемых параметров
            optimization_params = {}
            i = 0
            for key, param in self.dict_params.items(): 
                # Тип переменной
                type_func = param[0]
                # Приводим значения гиперпараметра к нужному типу
                num = type_func(X_next[i])
                optimization_params[key] = num
                i += 1
            # Кросс-валидация с сгенерированными параметрами 
            model = clone(self.model).set_params(**optimization_params)
            cv_results = cross_validate(model, X, y, scoring=self.scoring, cv=self.cv)
            # Оценить целевую функцию в новой точке
            Y_next = np.mean(cv_results["test_score"])
            # Добавляю известную точку к другим уже известным и заново учу гауссовский процесс
            X_sample = np.vstack((X_sample, X_next.reshape(1, -1))) 
            Y_sample = np.append(Y_sample, Y_next)
            gp.fit(X_sample, Y_sample)
        # Лучший score
        best_score = np.max(Y_sample)
        # Лучший набор параметров
        best_params_value = X_sample[np.argmax(Y_sample)]
        best_params = {}
        i = 0
        for key in self.dict_params: 
            best_params[key] = best_params_value[i]
            i += 1

        return best_score, best_params