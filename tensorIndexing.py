import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features),dtype=torch.float16)



print(x[0].shape) # first batch
print(x) # first batch

# to access the first feature for all of our examples in the batch:
""" print(x[:, 0]) # first feature (:) is basically the all thing.
print(x[:, 1]) # second feature (:) is basically the all thing.
 """
# to access the first feature for the first 10 examples in the batch:
""" print(x[8:10, 0]) # first feature of examples 4 to 10 """
# the indexing is like [start:stop) where start is inclusive and stop is exclusive

# to change the value of the first feature for the first 10 examples in the batch:
x[8:10, 0] = 100
print(x[8:10, 0]) # first feature of examples 4 to 10
print(x) # first batch

x = torch.arange(10)
print(x[(x <2) | (x > 8)])

print(x[x.remainder(2) == 0])

# other stuff:
print(torch.where(x > 5, x+12, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
# to check how many dims the tensor has:
print(x.ndimension())
# to check the shape of the tensor:
print(x.shape)
# to check the size of the tensor:
print(x.size())
# to check the number of elements in the tensor:
print(x.numel())


