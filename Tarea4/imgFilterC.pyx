import cython
import numpy as np

@cython.boundscheck(False)
cpdef imgFilterC(int[:, :] img, int[:, :] kernel):
    """
        Speed up de = 235.49
        Tiempo original = 1551.88
        Tiempo cython = 6.59
    """
    # declarar TODAS las vars
    cdef int rImg, cImg, rK, cK, R
    cdef int i, j, iK, jK, mult # vars fors
    cdef int[:, :] w, filtered 
    
    # lo mismo
    rImg, cImg = img.shape[0], img.shape[1]
    rK, cK = kernel.shape[0], kernel.shape[1] 

    R = rK//2
    filtered = np.zeros((rImg,cImg), dtype=np.int32)

    for i in range(R, rImg-R):# i filas j columnas
        for j in range(R, cImg-R):
            w = img[i-R:i+R+1,j-R:j+R+1]

            # t = np.multiply(w, kernel).sum()

            mult = 0
            for iK in range(0, rK):# i filas j columnas
                for jK in range(0, cK):
                    mult += w[iK,jK] * kernel[iK,jK]
            filtered[i,j] = mult
    return filtered