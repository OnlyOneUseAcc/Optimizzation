from scipy.sparse import csr_matrix
from Methods import Methods
import numpy as np


class BlockMatrix:
    def __init__(self, matrix):
        self.n = int(matrix.shape[0] / 2)
        self.matrix = [csr_matrix(matrix[0:self.n, 0:self.n]),
                       csr_matrix(matrix[0:self.n, self.n:]),
                       csr_matrix(matrix[self.n:, 0:self.n]),
                       csr_matrix(matrix[self.n:, self.n:])]

    def LU(self):
        inv_X = Methods.inverse(self.matrix[0])

        L = np.eye(self.n * 2)
        L[self.n:, 0: self.n] = self.matrix[2].toarray() * inv_X

        D = np.zeros((self.n * 2, self.n * 2))
        D[0:self.n, 0: self.n] = self.matrix[0].toarray()
        D[self.n:, self.n:] = self.matrix[3].toarray() - \
                              self.matrix[2].toarray() * inv_X * self.matrix[1].toarray()

        U = np.eye(self.n * 2)
        U[0: self.n, self.n:] = inv_X * self.matrix[1]

        return L, D, U
