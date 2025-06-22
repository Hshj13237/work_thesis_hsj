import torch


class trans_fnction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(y_act)
        return y_act

    @staticmethod
    def backward(ctx, grad_output):
        y_act, = ctx.saved_tensors
        grad_y_pre = grad_output.mul(y_act)
        grad_y_act = None
        return grad_y_pre, grad_y_act