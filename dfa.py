"""This module implements the DFA algorithm and injects Gaussian noise to the output of 
each MAC operation in the matrix-vector multiplication for calculating the gradient.
This code was inspired by https://github.com/lightonai/dfa-scales-to-modern-deep-learning"""

import numpy as np
import torch
from torch import nn


class DFAOutput(nn.Module):
    def __init__(self, dfa_layers, error_mean, error_std):
        super().__init__()
        self.dfa_layers = dfa_layers
        self.error_mean = error_mean
        self.error_std = error_std
        for i, dfa_layer in enumerate(self.dfa_layers):
            dfa_layer.hook_fun = self._generate_hook_fun(i)
        self.dfa_output_fun = self._DFAOutputFun.apply
        self.B_matrices = []  # Init during forward method
        self.B_e_list = [None for _ in enumerate(self.dfa_layers)]  # Updated by _DFAOutputFun
        self.init = False

    def forward(self, input):
        if not self.init:
            e_size = input.shape[1]  # Size of error vector
            for layer in self.dfa_layers:
                self.B_matrices.append(torch.rand(e_size, layer.size) * 2 - 1)
            self.init = True
        return self.dfa_output_fun(input, self)

    def _generate_hook_fun(self, layer_num):
        def _hook_fun(grad):
            return self.B_e_list[layer_num]

        return _hook_fun

    class _DFAOutputFun(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, dfa_context):
            ctx.dfa_context = dfa_context
            return input

        @staticmethod
        def backward(ctx, e_vector):
            dfa_context = ctx.dfa_context
            for i, B_Matrix in enumerate(dfa_context.B_matrices):
                B_e = e_vector @ B_Matrix
                noise = torch.max(torch.abs(e_vector)) * (
                    np.random.normal(dfa_context.error_mean, dfa_context.error_std, B_e.shape)
                )
                B_e += noise
                B_e /= B_e.shape[1] ** 0.5  # Normalize
                dfa_context.B_e_list[i] = B_e
            return e_vector, None


class DFALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hook_fun = None  # Updated by DFAOutput __init__ method
        self.size = None
        self.init = False

    def forward(self, input):
        if not self.init:
            self.size = input.shape[1]
            self.init = True
        if input.requires_grad:
            input.register_hook(self.hook_fun)
        return input
