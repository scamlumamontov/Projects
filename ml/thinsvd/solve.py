import numpy as np

def solve(A):
    known = None
    if A.shape[0] < A.shape[1]:
        A_temp = A @ A.T #Gives U
        known = "U"
    else:
        A_temp = A.T @ A #Gives V
        known = "V"

    #Eigen decomposition of real and symmetric matrix
    vals, vecs = np.linalg.eigh(A_temp)
    sing = np.sqrt(np.clip(vals, 0, None) ) #max(0, vecs_i)

    true_mat = None #Saving result
    sigma = sing.flatten() #Adding dimension (n,) -> (n, 1)

    if known == "U": 
        Vt = (vecs.T @ A) / sigma[:, None] #Divide by rows
        V = Vt.T
        true_mat = V
    else:
        AV = (A @ vecs)
        U = AV / sigma[None, :] #Divide by columns
        true_mat = U

    U_true, V_true = None, None
    if known == "V":
        U_true = true_mat
        Vt = (U_true.T @ A) / sigma[:, None] #Divide by rows
        V_true = Vt.T
    else:
        V_true = true_mat
        AV = A @ V_true
        U_true = AV / sigma[None, :] #Divide by columns

    return U_true @ np.diag(sigma) @ V_true.T