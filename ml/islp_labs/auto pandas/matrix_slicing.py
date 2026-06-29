import numpy as np
a = np.array([
    [1, 2, 3, 4, 5, 6],
    [5, 6, 7, 8, 9, 0],
    [1, 2, 3, 4, 5, 6]
    ])

#Extract row
print(a[:, 0])
print(a[:, [0, 1]])
#Extract column
print(a[[0]])

#Here goes a submatrix

print("subset:", a[[1, 2]][:, [1, 3]])

#Submatrix slicing
print("slicing:", a[0:2, 3:5])

#Boolean slicing
keep_rows = np.zeros(a.shape[0], bool) #1*(#a rows) matrix of zeroes
#bool to get zeroes of type boolean
keep_rows[1] = True #Make f[1] true to keep row 1
print("keeprows", keep_rows)
print(a[keep_rows])

keep_cols = np.zeros(a.shape[1], bool)
keep_cols[[1, 3]] = True
print(keep_cols)


test_subject = np.ix_(keep_rows, keep_cols) #Create some kind of shi with rows and columns, then find their cross product to create a mask

print("test", test_subject)
print(a[test_subject]) #submatrix

idx_mixed = np.ix_([0, 1], keep_cols) #mix
print(a[idx_mixed]) #subm