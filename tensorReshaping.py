import torch

x = torch.arange(10)

x_r = x.view(2,5)
#only acts on contiguous tensors, it needs to be contiguous. good performance, but not always possible

x_e = x.reshape(2,5)
#more flexible, but could have performance issues because it has to copy the data

""" print(x)
print(x_r) """
print(x_e)

#transposing:
y = x_r.t()
print(y)

#flattening it:
z = y.contiguous().view(1,10)
print(z)
u = y.contiguous().view(-1)
print(u)

