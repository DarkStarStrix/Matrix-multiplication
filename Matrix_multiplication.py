# matrix multiplication using numpy

import numpy 

# 2x2 matrix
# A = numpy.array([[1,2],[3,4]])
# print("Matrix A:")
# print(A)

# 2x2 matrix
# B = numpy.array([[5,6],[7,8]])
# print("Matrix B:")
# print(B)

# matrix multiplication
# C = numpy.dot(A,B)
# print("Matrix C:")
# print(C)

# matrix multiplication
# D = numpy.matmul(A,B)
# print("Matrix D:")
# print(D)

# matrix multiplication
# E = A@B
# print("Matrix E:")
# print(E)

# matrix multiplication
# F = numpy.einsum('ij,jk',A,B)
# print("Matrix F:")
# print(F)

# matrix multiplication using for loop
# G = numpy.zeros((2,2))
# for i in range(2):
 # for j in range(2):
        #for k in range(2):
           # G[i][j] += A[i][k]*B[k][j]
# print("Matrix G:")
# print(G)

# matrix multiplication using srassen algorithm use o(n^2.81) time complexity
# A = numpy.array([[1,2],[3,4]])
# B = numpy.array([[5,6],[7,8]])
# H = numpy.matmul(A,B)
# print("Matrix H:")
# print(H)

# matrix multiplcation using tensor decomposition use o(n^2.37) time complexity
# A = numpy.array([[1,2],[3,4]])
# B = numpy.array([[5,6],[7,8]])
# I = numpy.einsum('ij,jk',A,B)
# print("Matrix I:")
# print(I)

# matrix multiplication using numpy.tensordot
# A = numpy.array([[1,2],[3,4]])
# B = numpy.array([[5,6],[7,8]])
# J = numpy.tensordot(A,B,axes=1)
# print("Matrix J:")
# print(J)

# matrix multiplication using alpha tensor
# A = numpy.array([[1,2],[3,4]])
# B = numpy.array([[5,6],[7,8]])
# K = numpy.einsum('ij,jk->ik',A,B)
# print("Matrix K:")
# print(K)

# plot matrix multiplication time complexity using numpy
import matplotlib.pyplot as plt
import time

# matrix multiplication using numpy.dot
A = numpy.array([[1,2],[3,4]])
B = numpy.array([[5,6],[7,8]])
start = time.time()
C = numpy.dot(A,B)
end = time.time()
print("Matrix C:")
print(C)

# matrix multiplication using numpy.matmul
A = numpy.array([[1,2],[3,4]])
B = numpy.array([[5,6],[7,8]])
start = time.time()
D = numpy.matmul(A,B)
end = time.time()
print("Matrix D:")
print(D)

# plot the matrix multiplication time complexity using multi dimensional array using 3d plots

x = numpy.array([1,2,3,4,5,6,7,8,9,10])
y = numpy.array([0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100])
X,Y = numpy.meshgrid(x,y)
Z = numpy.array([[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,cmap='viridis',edgecolor='none')
ax.set_title('Surface plot')
plt.show()

# plot the matrix multiplication time complexity using multi dimensional array using 2d plots
x = numpy.array([1,2,3,4,5,6,7,8,9,10])
y = numpy.array([0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100])
X,Y = numpy.meshgrid(x,y)
Z = numpy.array([[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1
,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1
,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001,0.01
,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001,0.001
,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001,0.0001
,0.001,0.01,0.1,1,10,100],[0.0000001,0.000001,0.00001
,0.0001,0.001,0.01,0.1,1,10,100]])
plt.contour(X,Y,Z)
plt.show()



