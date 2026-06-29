import numpy as np

def forward_substitution(L, b): #LUx = b (Coefficients of REF)
    y = np.zeros(b.shape)
    n = L.shape[0]
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y

def backward_substitution(U, y): #Ly = b (Ux = y) (RREF)
    x = np.zeros(y.shape)
    n = U.shape[0]
    for i in np.arange(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def lu(A):
    P = np.zeros(A.shape)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is non-square")

    perm = np.arange(A.shape[0]) #Pivoting (greedy)
    l = 0
    while(l < len(perm)):
        mx, pos = 0, 0
        for i in range(l, len(perm)):
            val = abs(A[perm[i]][l])
            if val > mx:
                mx = val
                pos = i
        perm[l], perm[pos] = perm[pos], perm[l] #Greatest element on diagonal
        l += 1

    for i in enumerate(perm): #Pivot matrix
        P[i] = 1
    A_temp = (P @ A)

    L, U = np.zeros(A.shape), np.zeros(A.shape)

    eps = 1e-5
    for i in range(A.shape[0]):
        L[i, i] = 1
        U[i, i:] = A_temp[i, i:] #U - backward subst (transformation matrix to get to REF)
        L[i+1:, i] = A_temp[i+1:, i] / U[i, i] #L - Forward subst (Actual REF)

        if abs(U[i, i]) < eps:
            raise ValueError("Pivot is zero!")

        A_temp[i+1:, i+1:] -= np.outer(L[i+1:, i], U[i, i+1:]) #Step
    return P, L, U

def solve(P, L, U, b):
    b_perm = P @ b
    y = forward_substitution(L, b_perm)
    x = backward_substitution(U, y)
    return x