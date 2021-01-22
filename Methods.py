import numpy as np


class Methods:
    @staticmethod
    def get_from_matrix(csr, i, j):  # from 0
        element = 0
        start = csr.indptr[i]
        end = csr.indptr[i + 1]

        for k in range(start, end, 1):
            if (csr.indices[k] == j):
                element = csr.data[k]
                break

        return element

    @staticmethod
    def get_lu(csr):
        lu = np.matrix(np.zeros(csr.shape))
        for k in range(csr.shape[0]):
            for j in range(k, csr.shape[0]):
                lu[k, j] = Methods.get_from_matrix(csr, k, j) - lu[k, :k] * lu[:k, j]

            for i in range(k + 1, csr.shape[0]):
                lu[i, k] = (Methods.get_from_matrix(csr, i, k) - lu[i, : k] * lu[: k, k]) / lu[k, k]

        return lu

    @staticmethod
    def solve(A, b):
        lu = Methods.get_lu(A)
        y = np.matrix(np.zeros([lu.shape[0], 1]))

        for i in range(0, y.shape[0]):
            y[i, 0] = b[i, 0] - lu[i, :i] * y[:i]

        x = np.matrix(np.zeros([lu.shape[0], 1]))
        for i in range(1, y.shape[0] + 1):
            x[-i, 0] = (y[-i] - lu[-i, -i:] * x[-i:, 0]) / lu[-i, -i]

        return x

    @staticmethod
    def inverse(csr):
        inv = np.zeros(csr.shape)
        for n in range(csr.shape[0]):
            b_i = np.array(np.zeros([csr.shape[0], 1]))
            b_i[n][0] = 1
            inv[:, n] = Methods.solve(csr, b_i).reshape(1, -1)
        return inv

    


