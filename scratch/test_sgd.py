import torch

# PyTorch
param = torch.tensor([1.0], requires_grad=True)
grad = torch.tensor([0.1])
param.backward(grad)

optimizer = torch.optim.SGD([param], lr=0.1, momentum=0.9, weight_decay=1e-4)

# step 1
print("PyTorch init:", param.data.item())
optimizer.step()
print("PyTorch step 1:", param.data.item())

param.grad = None
param.backward(torch.tensor([0.2]))
optimizer.step()
print("PyTorch step 2:", param.data.item())

# TunX
tunx_param = 1.0
tunx_velocity = 0.0

lr = 0.1
momentum = 0.9
weight_decay = 1e-4

# step 1
grad_1 = 0.1
grad_1_wd = grad_1 + weight_decay * tunx_param
tunx_velocity = momentum * tunx_velocity - lr * grad_1_wd
tunx_param = tunx_param + tunx_velocity
print("TunX step 1:", tunx_param)

# step 2
grad_2 = 0.2
grad_2_wd = grad_2 + weight_decay * tunx_param
tunx_velocity = momentum * tunx_velocity - lr * grad_2_wd
tunx_param = tunx_param + tunx_velocity
print("TunX step 2:", tunx_param)

