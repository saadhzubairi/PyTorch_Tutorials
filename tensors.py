import torch

# Tensor is a multi-dimensional matrix containing elements of a single data type. Here are some basics:

# Create a 2x3 tensor with random values
x = torch.rand(2, 3)
# print(x)


# now we setup the device to be used:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# always use cuda if available, otherwise use cpu


# print(device)

# A tensor with a list inside a list, 2 rows and 3 columns:
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32,device=device, requires_grad=True)

# the device is set to the device we defined above, so it will be using the gpu if available, otherwise it will use the cpu

# the data type is set to float32, which is a 32-bit floating point number

# the auto_grad is set to true, so we can use the gradients to update the weights, which means we can use the backpropagation algorithm to update the weights of the neural network model. it's useful for training the model and making predictions. the math of it is a bit complicated, but it's a way to optimize the model to make better predictions.

""" print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad) """

#other common initialization methods include:
x0 = torch.empty(size=(3,3)) # creates an empty tensor
x1 = torch.zeros((3,3)) # creates a tensor with zeros
x2 = torch.rand((3,3)) # creates a tensor with random values
x3 = torch.ones((3,3)) # creates a tensor with ones
x4 = torch.eye(5,5) # creates an identity matrix
x5 = torch.arange(start=0, end=5, step=1) # creates a tensor with values from 0 to 4
# to create a 2d tensor with values from 0 to 4 in every row, we can use the following code:
x5 = torch.arange(start=0, end=5, step=1).view(1,5).repeat(3,1)
x6 = torch.linspace(start=0.1, end=1, steps=10) # creates a tensor with 10 values from 0.1 to 1
x7 = torch.empty(size=(1,5)).normal_(mean=0, std=1) # creates a tensor with random values from a normal distribution with mean 0 and std 1
x8 = torch.empty(size=(1,5)).uniform_(0,1) # creates a tensor with random values from a uniform distribution between 0 and 1
x9 = torch.diag(torch.ones(3)) # creates a diagonal matrix with ones on the diagonal

# print(f"0: {x0}")
# print(f"1: {x1}")
# print(f"2: {x2}")
# print(f"3: {x3}")
# print(f"4: {x4}")
# print(f"5: {x5}")
# print(f"6: {x6}")
# print(f"7: {x7}")
# print(f"8: {x8}")
# print(f"9: {x9}")

# Now learning some converssion between tensor types (int, float, double, etc)

t = torch.arange(4) # default is int64
""" print(f"t {t}, t.dtype {t.dtype}")
print(f"t.bool() {t.bool()}") # boolean
print(f"t.short() {t.short()}") # int16
print(f"t.long() {t.long()}") # int64
print(f"t.half() {t.half()}") # float16
print(f"t.float() {t.float()}") # float32
print(f"t.double() {t.double()}") # float64

 """
 
 # how to convert tensor to numpy array and vice versa
 
import numpy as np
 
np_array = np.ones((5,5))
tensor = torch.from_numpy(np_array)

print(f"tensor {tensor}")
print(f"np_array {np_array}")

# now we can convert the tensor back to a numpy array
np_array_back = tensor.numpy()

print(f"np_array_back {np_array_back}")