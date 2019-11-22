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

matrixs = np.array([matrixK, matrixA, matrixD])
noised_matrixs = [noise_matrixK, noise_matrixA, noise_matrixD]

def train(matrixs):
    n,m = matrixs.shape
    W = np.zeros((m,m))
    for p in matrixs:
        W = W + np.outer(p,p)
    W[np.diag_indices(m)] = 0
    return W

def recall(W, matrixs, steps):
    for s in range (steps):        
        matrixs = np.array(np.sign(np.dot(matrixs,W)))   
    print(matrixs)    
    return matrixs

def display(patterns):
    for pattern in patterns:
        pl.imshow(pattern.reshape((9,7)),cmap=pl.cm.binary, interpolation='nearest')
        pl.show()
            
display(matrixs)
display(noised_matrixs)
result_matrixs = recall(train(matrixs),noised_matrixs,2)
display(result_matrixs)