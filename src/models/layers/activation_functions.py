# import torch
# from efficientnet_pytorch.model import MemoryEfficientSwish
# from torch import nn
# import torch.nn.functional as F
#
# class MishFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x = ctx.saved_variables[0]
#         sigmoid = torch.sigmoid(x)
#         tanh_sp = torch.tanh(F.softplus(x))
#         return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))
#
# class MemoryEfficientMish(nn.Module):
#     def forward(self, x):
#         return MishFunction.apply(x)
#
# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x * (torch.tanh(F.softplus(x)))
