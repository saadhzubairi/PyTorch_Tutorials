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

