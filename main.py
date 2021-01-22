from Methods import Methods
from scipy.sparse import csr_matrix
from TestFunctons import MatrixTest
from BlockMatrix import BlockMatrix
import numpy as np


MatrixTest.show_error_d(15)
MatrixTest.show_error_h(15)

a = BlockMatrix(np.array([
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [14,15,16,17]]))

print(a.LU())



