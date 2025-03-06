# Tensor math and comparison operations

import torch

x = torch.tensor([1, 2, 3],dtype=torch.short)
y = torch.tensor([4, 5, 6],dtype=torch.short)

# addition

z1 = torch.empty(3)
torch.add(x, y, out=z1)
# print(z1)

z2 = torch.add(x, y)
# print(z2)

z = x + y
# print(z)

# subtraction

z = torch.empty(3)
torch.sub(x, y, out=z)
# print(z)

z = torch.sub(x, y)
# print(z)

z = x - y
# print(z)

# division

z = torch.true_divide(x, y)
# print(z)

#what true divide does is that it converts the values to float before dividing them, and does element wise division if they are of equal shape. 

z = torch.true_divide(x, 3)
# print(z)

# in-place operations
t = torch.zeros(3)
t.add_(x)
# print(t)
t += x

# not exactly in-place:
t = t + x

#exponentiation

z = x.pow(2)
# print(z)

#or 
z = x ** 2
# print(z)

#simple comparison
z = x > 0
# print(z)

# matrix operations:

x1 = torch.rand((2,5))
x2 = torch.rand((5,3))

x3 = torch.mm(x1, x2) # matrix multiplication
x3 = x1.mm(x2) # matrix multiplication

print(x3)

# matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp)
matrix_exp.matrix_power(3)
print(matrix_exp)

# element wise multiplication 
z = x * y
print(z)
z = x.mul(y)
print(z)
z = x.mul_(y)
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m)) # 32 matrices of size 10x20
tensor2 = torch.rand((batch, m, p)) # 32 matrices of size 20x30

out = torch.bmm(tensor1, tensor2) # 32 matrices of size 10x30

#to access second row of the first matrix of the batch:
first = out[0][1]
print(first)



