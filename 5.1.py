import numpy as np
import pylab as pl

#Original matrix

matrixK = np.array([1, -1, -1, -1, -1, 1, -1,
                    1, -1, -1, -1, 1, -1, -1,
                    1, -1, -1, 1, -1, -1, -1,
                    1, -1, 1, -1, -1, -1, -1,
                    1, 1, -1, -1, -1, -1, -1,
                    1, -1, 1, -1, -1, -1, -1,
                    1, -1, -1, 1, -1, -1, -1,
                    1, -1, -1, -1, 1, -1, -1,
                    1, -1, -1, -1, -1, 1, -1])

matrixA = np.array([-1, -1, -1, 1, -1, -1, -1,
                    -1, -1, 1, -1, 1, -1, -1,
                    -1, 1, -1, -1, -1, 1, -1,
                    1, -1, -1, -1, -1, -1, 1,
                    1, -1, -1, -1, -1, -1, 1,
                    1, 1, 1, 1, 1, 1, 1,
                    1, -1, -1, -1, -1, -1, 1,
                    1, -1, -1, -1, -1, -1, 1,
                    1, -1, -1, -1, -1, -1, 1])

matrixD = np.array([1, 1, -1, -1, -1, -1, -1,
                    1, -1, 1, -1, -1, -1, -1,
                    1, -1, -1, 1, -1, -1, -1,
                    1, -1, -1, -1, 1, -1, -1,
                    1, -1, -1, -1, -1, 1, -1,
                    1, -1, -1, -1, 1, -1, -1,
                    1, -1, -1, 1, -1, -1, -1,
                    1, -1, 1, -1, -1, -1, -1,
                    1, 1, -1, -1, -1, -1, -1])

#Noised matrix 

noise_matrixK = np.array([1, -1, -1, -1, -1, 1, 1,
                          1, 1, -1, -1, 1, -1, 1,
                          1, -1, -1, 1, 1, -1, -1,
                          1, -1, 1, -1, 1, -1, -1,
                          1, 1, -1, -1, 1, -1, 1,
                          1, -1, 1, -1, 1, -1, -1,
                          1, -1, -1, 1, 1, -1, 1,
                          1, -1, -1, -1, 1, -1, -1,
                          1, 1, -1, -1, -1, 1, -1])

noise_matrixA = np.array([-1, -1, 1, 1, 1, -1, 1,
                          -1, -1, 1, 1, 1, -1, -1,
                          -1, 1, -1, 1, -1, 1, 1,
                          1, 1, -1, 1, 1, -1, -1,
                          1, -1, -1, 1, 1, -1, 1,
                          1, 1, -1, 1, 1, 1, 1,
                          1, -1, -1, -1, -1, -1, 1,
                          1, -1, -1, -1, -1, 1, 1,
                          1, 1, -1, 1, -1, -1, 1])

noise_matrixD = np.array([1, 1, 1, -1, -1, -1, -1,
                          1, -1, 1, -1, -1, 1, -1,
                          1, -1, -1, 1, 1, -1, -1,
                          -1, 1, -1, -1, 1, -1, -1,
                          1, -1, 1, -1, -1, 1, -1,
                          1, -1, -1, -1, 1, -1, -1,
                          1, -1, 1, -1, -1, -1, -1,
                          1, -1, 1, -1, -1, -1, -1,
                          -1, 1, -1, -1, -1, -1, -1])

#ASCII Binary Code

ascii_K = np.array([1,-1,-1,1,-1,1,1])
ascii_A = np.array([1,-1,-1,-1,-1,-1,1])
ascii_D = np.array([1,-1,-1,-1,1,-1,-1])

#Arrays with data
matrixs = np.array([matrixK, matrixA, matrixD])
noised_matrixs = np.array([noise_matrixK, noise_matrixA, noise_matrixD])
ascii_code = np.array([ascii_K, ascii_A, ascii_D])

    
def train(matrixs, ascii_code):
    W = np.zeros((63,7))
    for m, a in zip(matrixs,ascii_code):
        W = W + np.outer(m,a)
    W[np.diag_indices(7)] = 0
    return W
    
        

def recall(W, matrixs, steps):
        result = np.array(np.sign(np.dot(matrixs, W)))
    print(result)    
    return result    


def display(matrixs):
    for matrix in matrixs:
        pl.imshow(matrix.reshape((9,7)),cmap=pl.cm.binary, interpolation='nearest')
        pl.show()
            
display(matrixs)
display(noised_matrixs)
result_ascii_code = recall(train(matrixs, ascii_code), noised_matrixs, 2)
print(result_ascii_code)

