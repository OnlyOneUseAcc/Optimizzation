import numpy as np
import random
from scipy.sparse import csr_matrix
from Methods import Methods
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


class MatrixTest:
    @staticmethod
    def hilbert(n):
        return csr_matrix(np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)]))

    @staticmethod
    def diagonal(n, k):
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    result[i][j] = random.randint(-4, 0)
        for i in range(n):
            result[i][i] = - sum(result[i, :]) + 10 ** (-k)
        return csr_matrix(result)

    @staticmethod
    def F(csr_A):
        n = csr_A.shape[0]
        return csr_A.toarray().dot(np.arange(1, n + 1, 1))

    @staticmethod
    def show_error_d(k):
        errors = []
        k_list = []
        for i in range(2, k + 1):
            A = MatrixTest.diagonal(i, i)
            F = MatrixTest.F(csr_matrix(A))
            predict = Methods.solve(A, F.reshape(i, -1))
            result = np.arange(1, A.shape[0] + 1, 1)
            k_list.append(i)
            errors.append(mean_absolute_error(result, predict))
        plt.plot(k_list, errors)
        plt.ylabel('Error')
        plt.xlabel('k')
        plt.show()

    @staticmethod
    def show_error_h(n):
        errors = []
        k_list = []
        for i in range(2, n + 1):
            A = MatrixTest.hilbert(i)
            F = MatrixTest.F(csr_matrix(A))
            predict = Methods.solve(A, F.reshape(i, -1))
            result = np.arange(1, A.shape[0] + 1, 1)
            k_list.append(i)
            errors.append(mean_absolute_error(result, predict))
        plt.plot(k_list, errors)
        plt.ylabel('Error')
        plt.xlabel('k')
        plt.show()






