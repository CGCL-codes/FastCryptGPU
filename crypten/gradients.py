#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce

from crypten.cuda import CUDALongTensor

from crypten.mpc import primitives

import crypten
import torch
import torch.nn.functional as F
from crypten.common.util import im2col_indices, col2im_indices
import crypten.communicator as comm
import time

# registry that maps function names to AutogradFunctions:
FUNCTION_REGISTRY = {}


def register_function(name):
    """Decorator that registers a new autograd function."""

    def register_function_cls(cls):
        """Function performing the actual registration."""
        if name in FUNCTION_REGISTRY:
            raise ValueError("Cannot register duplicate function ({})".format(name))
        if not issubclass(cls, AutogradFunction):
            raise ValueError(
                "Function (%s: %s) must extend AutogradFunction" % (name, cls.__name__)
            )
        cls.name = name
        FUNCTION_REGISTRY[name] = cls
        return cls

    return register_function_cls


def timer_conv(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_conv += run_time
        return value

    return wrapper_timer


def timer_pool(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_pool += run_time
        return value

    return wrapper_timer


def timer_relu(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_relu += run_time
        return value

    return wrapper_timer


def timer_matmul(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_matmul += run_time
        return value

    return wrapper_timer


def timer_softmax(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_softmax += run_time
        return value

    return wrapper_timer


def get_grad_fn(name):
    """
    Returns gradient function for the CrypTen function with the specified name.
    """
    if name in FUNCTION_REGISTRY:
        return FUNCTION_REGISTRY[name]
    else:
        return None


def _ensure_tensor(input):
    """
    Converts scalars in inputs to correct tensor type.
    """
    if isinstance(input, (int, float)):
        input = torch.tensor(input)
    return input


def _inverse_broadcast(grad_output, input_size):
    """
    Performs the inverse operation of a broadcast.
    """

    # special case where input was a scalar:
    if input_size == torch.Size():
        return grad_output.sum()

    # remove leading dimensions:
    while grad_output.dim() > len(input_size):
        grad_output = grad_output.sum(0, keepdim=False)
    assert grad_output.dim() == len(input_size), "cannot perform inverse broadcast"

    # perform accumulation across broadcast dimensions:
    for dim in range(grad_output.dim()):
        if input_size[dim] == 1 and grad_output.size(dim) > 1:
            grad_output = grad_output.sum(dim, keepdim=True)
    return grad_output


class AutogradContext(object):
    """
    Object that can be used by AutogradFunction for saving context information.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.context = []
        self.non_differentiable = []

    def save_for_backward(self, value):
        self.context.append(value)

    def save_multiple_for_backward(self, values):
        for value in values:
            self.save_for_backward(value)

    def mark_non_differentiable(self, non_differentiable):
        if not isinstance(non_differentiable, list):
            non_differentiable = [non_differentiable]
        self.non_differentiable.extend(id(x) for x in non_differentiable)

    def is_differentiable(self, tensor):
        return id(tensor) not in self.non_differentiable

    @property
    def saved_tensors(self):
        return self.context


def timer_rule(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_relu += (run_time)
        return value

    return wrapper_timer

class AutogradFunction(object):
    """
    Base implementation of a function that supports autograd.
    """

    @staticmethod
    def forward(ctx, input):
        raise NotImplementedError("Forward function not implemented.")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward function not implemented.")

    def __str__(self):
        if hasattr(self, "name"):
            return self.name


@register_function("t")
class AutogradT(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.t()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.t()


@register_function("transpose")
class AutogradTranspose(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim1, dim2):
        ctx.save_multiple_for_backward((dim1, dim2))
        return input.transpose(dim1, dim2)

    @staticmethod
    def backward(ctx, grad_output):
        dim1, dim2 = ctx.saved_tensors
        return grad_output.transpose(dim2, dim1)


@register_function("flip")
class AutogradFlip(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dims):
        ctx.save_for_backward(dims)
        return input.flip(dims)

    @staticmethod
    def backward(ctx, grad_output):
        (dims,) = ctx.saved_tensors
        return grad_output.flip(dims)


@register_function("clone")
class AutogradClone(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


@register_function("cat")
class AutogradCat(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=0):
        split_sections = [t.size(dim) for t in input]
        ctx.save_multiple_for_backward((dim, split_sections))
        return crypten.cat(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim, split_sections = ctx.saved_tensors
        return grad_output.split(split_sections, dim=dim)


@register_function("stack")
class AutogradStack(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=0):
        ctx.save_for_backward(dim)
        return crypten.stack(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return grad_output.unbind(dim=dim)


@register_function("view")
class AutogradView(AutogradFunction):
    @staticmethod
    def forward(ctx, input, *size):
        ctx.save_for_backward(input.size())
        return input.view(*size)

    @staticmethod
    def backward(ctx, grad_output):
        (input_size,) = ctx.saved_tensors
        return grad_output.view(input_size)


@register_function("reshape")
class AutogradReshape(AutogradFunction):
    @staticmethod
    def forward(ctx, input, shape):
        ctx.save_for_backward(input.size())
        return input.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        (size,) = ctx.saved_tensors
        return grad_output.reshape(size)


@register_function("flatten")
class AutogradFlatten(AutogradFunction):
    @staticmethod
    def forward(ctx, input, start_dim=0, end_dim=-1):
        ctx.save_for_backward(input.size())
        return input.flatten(start_dim=start_dim, end_dim=end_dim)

    @staticmethod
    def backward(ctx, grad_output):
        (size,) = ctx.saved_tensors
        return grad_output.reshape(size)


@register_function("squeeze")
class AutogradSqueeze(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):

        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to squeeze in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to squeeze in args

        # perform the actual squeeze:
        output = input.squeeze() if dim is None else input.squeeze(dim)

        # keep correct dimensions for backward pass:
        if dim is None:
            dims = [idx for idx, sz in enumerate(input.size()) if sz == 1]
        else:
            # Squeezeing non singleton dimensions is a no-op:
            dims = [dim] if input.size(dim) == 1 else []
        ctx.save_for_backward(dims)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (dims,) = ctx.saved_tensors
        grad_input = grad_output
        for dim in dims:
            grad_input = grad_input.unsqueeze(dim)
        return grad_input


@register_function("unsqueeze")
class AutogradUnsqueeze(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(dim)
        return input.unsqueeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return grad_output.squeeze(dim)


@register_function("__getitem__")
class AutogradGetItem(AutogradFunction):
    @staticmethod
    def forward(ctx, input, index):
        ctx.save_multiple_for_backward([input.size(), index])
        return input[index]

    @staticmethod
    def backward(ctx, grad_output):
        size, index = ctx.saved_tensors
        grad = grad_output.new(torch.zeros(size))
        grad[index] = grad_output
        return grad


@register_function("neg")
class AutogradNeg(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.neg()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


@register_function("add")
class AutogradAdd(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input.size(), other.size()])
        return input.add(other)

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1),
            _inverse_broadcast(grad_output.clone(), input_size2),
        )


@register_function("sub")
class AutogradSub(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input.size(), other.size()])
        return input.sub(other)

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1),
            _inverse_broadcast(grad_output.clone(), input_size2).neg(),
        )


@register_function("__rsub__")
class AutogradRSub(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input.size(), other.size()])
        return (-input).add(other)

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1).neg(),
            _inverse_broadcast(grad_output.clone(), input_size2),
        )


@register_function("mul")
class AutogradMul(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input, other])
        return input.mul(other)

    @staticmethod
    def backward(ctx, grad_output):
        self_, other = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.mul(other), self_.size()),
            _inverse_broadcast(grad_output.mul(self_), other.size()),
        )


@register_function("matmul")
class AutogradMatMul(AutogradFunction):
    @staticmethod
    @timer_matmul
    def forward(ctx, input, other):
        ctx.save_multiple_for_backward([input, other])
        return input.matmul(other)

    @staticmethod
    @timer_matmul
    def backward(ctx, grad_output):
        self_, other = ctx.saved_tensors

        # Cache sizes for invers_broadcast
        self_size = self_.size()
        other_size = other.size()

        # Deal with vectors that are represented by a
        # < 2 dimensional tensor
        if self_.dim() < 2:
            self_ = self_.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)

        if other.dim() < 2:
            other = other.unsqueeze(1)
            grad_output = grad_output.unsqueeze(1)

        # Compute gradients
        self_grad = grad_output.matmul(other.transpose(-2, -1))
        other_grad = self_.transpose(-2, -1).matmul(grad_output)

        # Fix gradient sizes for vector inputs
        if len(self_size) < 2:
            self_grad = self_grad.squeeze()
            if self_grad.dim() < 1:
                self_grad = self_grad.unsqueeze(0)

        if len(other_size) < 2:
            other_grad = other_grad.squeeze()
            if other_grad.dim() < 1:
                other_grad = other_grad.unsqueeze(0)

        return (
            _inverse_broadcast(self_grad, self_size),
            _inverse_broadcast(other_grad, other_size),
        )


@register_function("div")
class AutogradDiv(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        if crypten.is_encrypted_tensor(other):
            other_reciprocal = other.reciprocal()
            ctx.save_multiple_for_backward([input, other_reciprocal])
            return input.mul(other_reciprocal)
        else:
            ctx.save_multiple_for_backward([input.size(), other])
            return input.div(other)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors

        # saved is a list of [input, other_reciprocal]
        if crypten.is_encrypted_tensor(saved[1]):
            input, other_reciprocal = saved
            grad_input = other_reciprocal.mul(grad_output)
            grad_other = other_reciprocal.square().mul(input).mul(grad_output).neg()
            return (
                _inverse_broadcast(grad_input, input.size()),
                _inverse_broadcast(grad_other, other_reciprocal.size()),
            )
        # saved is a public tensor or scalar
        else:
            input_size, other = saved
            grad_input = grad_output.div(other)
            if torch.is_tensor(other):
                return _inverse_broadcast(grad_input, input_size)
            else:
                return grad_input


@register_function("__rtruediv__")
class AutogradRDiv(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        reciprocal = input.reciprocal()
        ctx.save_multiple_for_backward([reciprocal, other])
        return reciprocal.mul(other)

    @staticmethod
    def backward(ctx, grad_output):
        reciprocal, other = ctx.saved_tensors
        grad_input = reciprocal.square().mul(other).mul(grad_output).neg()
        grad_input = _inverse_broadcast(grad_input, reciprocal.size())

        if torch.is_tensor(other) or crypten.is_encrypted_tensor(other):
            grad_other = reciprocal.mul(grad_output)
            grad_other = _inverse_broadcast(grad_other, other.size())
            return (grad_input, grad_other)
        else:
            return grad_input


@register_function("pow")
class AutogradPow(AutogradFunction):
    @staticmethod
    def forward(ctx, input, power):
        ctx.save_multiple_for_backward([input, power])
        return input.pow(power)

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.saved_tensors
        return input.pow(power - 1.0).mul_(power).mul_(grad_output)


@register_function("pos_pow")
class AutogradPosPow(AutogradFunction):
    @staticmethod
    def forward(ctx, input, power):
        if isinstance(power, int) or (isinstance(power, float) and int(power) == power):
            ctx.save_multiple_for_backward([input, power])
            return input.pow(power)
        else:
            log_input = input.log()
            ctx.save_multiple_for_backward([log_input, power])
            return log_input.mul(power).exp()

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.saved_tensors
        if isinstance(power, int) or (isinstance(power, float) and int(power) == power):
            return input.pow(power - 1.0).mul_(power).mul_(grad_output)
        else:
            return input.mul(power - 1.0).mul_(power).exp().mul(grad_output)


@register_function("square")
class AutogradSquare(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.square()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output.mul(input.mul(2.0))


@register_function("sqrt")
class AutogradSqrt(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        output = input.sqrt()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return grad_output.div(output.mul_(2.0))


@register_function("exp")
class AutogradExp(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        output = input.exp()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return output.mul(grad_output)


@register_function("log")
class AutogradLog(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.log()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output.div(input)


@register_function("reciprocal")
class AutogradReciprocal(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        reciprocal = input.reciprocal()
        ctx.save_for_backward(reciprocal)
        return reciprocal

    @staticmethod
    def backward(ctx, grad_output):
        (reciprocal,) = ctx.saved_tensors
        return grad_output.neg().mul_(reciprocal).mul_(reciprocal)


@register_function("dot")
class AutogradDot(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_multiple_for_backward([input, other])
        return input.dot(other)

    @staticmethod
    def backward(ctx, grad_output):
        self_, other = ctx.saved_tensors
        return (grad_output.mul(other), grad_output.mul(self_))


@register_function("ger")
class AutogradGer(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_multiple_for_backward([input, other])
        return input.ger(other)

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        return (grad_output.matmul(other), input.matmul(grad_output))


@register_function("abs")
class AutogradAbs(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        sign = input.sign()
        ctx.save_for_backward(sign)
        return input.mul(sign)

    @staticmethod
    def backward(ctx, grad_output):
        (sign,) = ctx.saved_tensors
        return grad_output.mul(sign.mul_(2.0).sub_(1.0))


@register_function("sign")
class AutogradSign(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.sub(grad_output)


@register_function("sum")
class AutogradSum(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):

        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to sum over in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to sum over in args
        keepdim = kwargs.get("keepdim", False)

        # compute sum:
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.sum(dim, keepdim=keepdim) if dim is not None else input.sum()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors

        # Handle special case where input is 0-dimensional
        if len(input_size) == 0:
            return grad_output

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(torch.ones(input_size))


@register_function("cumsum")
class AutogradCumsum(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(dim)
        return input.cumsum(dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return grad_output.flip(dim).cumsum(dim).flip(dim)


@register_function("trace")
class AutogradTrace(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.size()[0])
        return input.trace()

    @staticmethod
    def backward(ctx, grad_output):
        (size,) = ctx.saved_tensors
        return grad_output.new(torch.eye(size)).mul_(grad_output)


@register_function("min")
class AutogradMin(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):

        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to min over in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to min over in args
        keepdim = kwargs.get("keepdim", False)
        one_hot = kwargs.get("one_hot", True)

        # find minimum value (and corresponding argmin):
        if dim is None:
            argmin = input.argmin(one_hot=one_hot)
            min = input.mul(argmin).sum()
        else:
            min, argmin = input.min(dim, keepdim=keepdim, one_hot=one_hot)

        # save context and return:
        ctx.save_multiple_for_backward((dim, keepdim, argmin, one_hot))
        if dim is None:
            return min
        else:
            ctx.mark_non_differentiable(argmin)
            return min, argmin

    @staticmethod
    def backward(ctx, grad_output):
        dim, keepdim, argmin, one_hot = ctx.saved_tensors
        assert one_hot, (
            "cannot backpropagate through min layer that does not"
            "use one-hot representation because private indexing is unsupported"
        )
        # Handle special case where input is 0-dimensional
        if len(argmin.size()) == 0:
            return grad_output

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(argmin)


@register_function("max")
class AutogradMax(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):

        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to max over in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to max over in args
        keepdim = kwargs.get("keepdim", False)
        one_hot = kwargs.get("one_hot", True)
        # find maximum value (and corresponding argmax):
        if dim is None:
            shape = input.size()
            input_flat = input.flatten()
            max, argmax = input_flat.max(0, **kwargs)
            argmax = argmax.reshape(shape)
        else:
            max, argmax = input.max(dim, **kwargs)

        # save context and return:
        ctx.save_multiple_for_backward((dim, keepdim, argmax, one_hot))
        if dim is None:
            return max
        else:
            ctx.mark_non_differentiable(argmax)
            return max, argmax

    @staticmethod
    def backward(ctx, grad_output):
        dim, keepdim, argmax, one_hot = ctx.saved_tensors
        assert one_hot, (
            "cannot backpropagate through max layer that does not"
            "use one-hot representation because private indexing is unsupported"
        )
        # Handle special case where input is 0-dimensional
        if len(argmax.size()) == 0:
            return grad_output

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(argmax)


@register_function("pad")
class AutogradPad(AutogradFunction):
    @staticmethod
    def forward(ctx, input, padding, value=0.0, mode="constant"):
        ctx.save_for_backward(padding)
        output = input.pad(padding, value=value, mode=mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (padding,) = ctx.saved_tensors
        for idx in range(0, len(padding), 2):
            dim = grad_output.dim() - (idx // 2) - 1
            start = padding[idx]
            end = grad_output.size(dim) - padding[idx + 1] - padding[idx]
            grad_output = grad_output.narrow(dim, start, end)
        return grad_output


@register_function("avg_pool2d")
class AutogradAvgPool2D(AutogradFunction):
    @staticmethod
    @timer_pool
    def forward(ctx, input, kernel_size, padding=0, stride=None):
        # print("come gradients.py AutogradAvgPool2D ")
        # preprocess inputs:
        if stride is None:
            stride = kernel_size
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        if isinstance(kernel_size, (int, float)):
            kernel_size = (kernel_size, kernel_size)

        # perform average pooling:
        output = input.avg_pool2d(kernel_size, padding=padding, stride=stride)

        # store information for backward pass:
        ctx.save_multiple_for_backward(
            (input.shape, kernel_size, padding, stride)
        )
        return output

    @staticmethod
    @timer_pool
    def backward(ctx, grad_output):
        input_size, kernel_size, padding, stride = ctx.saved_tensors
        bs, c = input_size[0], input_size[1]
        ky, kx = kernel_size

        assert grad_output.size(1) == c, "wrong number of output channels"
        assert grad_output.size(0) == bs, "wrong batch size"

        grad_out_reshaped = grad_output.permute(1, 2, 3, 0).reshape(c, 1, -1)
        kernel = crypten.cryptensor(torch.ones((c, ky * kx, 1), device=grad_output.device) / (ky * kx))
        grad_input_fold = kernel.matmul(grad_out_reshaped)
        grad_input = col2im_indices(grad_input_fold, input_size, ky, kx, padding=padding, stride=stride)

        return grad_input


@register_function("conv1d")
class AutogradConv1D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, kernel, padding=0, stride=1):
        ctx.save_multiple_for_backward((input, kernel, padding, stride))
        return input.conv1d(kernel, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        # get input, kernel, and sizes:
        input, kernel, padding, stride = ctx.saved_tensors
        batch_size = input.size(0)
        out_channels, in_channels, kernel_size = kernel.size()
        assert input.size(1) == in_channels, "wrong number of input channels"
        assert grad_output.size(1) == out_channels, "wrong number of output channels"
        assert grad_output.size(0) == batch_size, "wrong batch size"

        # compute gradient with respect to input:
        output_padding = torch.nn.grad._grad_input_padding(
            grad_output, input.size(), (stride,), (padding,), (kernel_size,)
        )
        grad_input = grad_output.conv_transpose1d(
            kernel, stride=stride, padding=padding, output_padding=output_padding
        )

        # compute gradient with respect to kernel:
        grad_output = grad_output.repeat(1, in_channels, 1)
        grad_output = grad_output.view(
            grad_output.size(0) * grad_output.size(1), 1, grad_output.size(2)
        )
        input = input.view(1, input.size(0) * input.size(1), input.size(2))
        grad_kernel = input.conv1d(
            grad_output,
            padding=padding,
            dilation=stride,
            groups=in_channels * batch_size,
        )
        grad_kernel = grad_kernel.view(
            batch_size, grad_kernel.size(1) // batch_size, grad_kernel.size(2)
        )
        grad_kernel = (
            grad_kernel.sum(dim=0)
                .view(in_channels, out_channels, grad_kernel.size(2))
                .transpose(0, 1)
                .narrow(2, 0, kernel_size)
        )
        return (grad_input, grad_kernel)


@register_function("conv2d")
class AutogradConv2D(AutogradFunction):
    @staticmethod
    @timer_conv
    def forward(ctx, input, kernel, padding=0, stride=1, groups=1):
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        ctx.save_multiple_for_backward((input, kernel, padding, stride, groups))
        return input.conv2d(kernel, padding=padding, stride=stride, groups=groups)

    @staticmethod
    @timer_conv
    def backward(ctx, grad_output):
        input, kernel, padding, stride, groups = ctx.saved_tensors
        batch_size = input.size(0)
        out_channels, in_channels, kernel_size_y, kernel_size_x = kernel.size()

        assert input.size(1) == in_channels, "wrong number of input channels"
        assert grad_output.size(1) == out_channels, "wrong number of output channels"
        assert grad_output.size(0) == batch_size, "wrong batch size"

        grad_out_reshaped = grad_output.permute(1, 2, 3, 0).reshape(out_channels, -1)

        input_fold = im2col_indices(input, kernel_size_y, kernel_size_x, padding=padding, stride=stride)
        grad_kernel = grad_out_reshaped.matmul(input_fold.transpose(0, 1))
        grad_kernel = grad_kernel.view(kernel.shape)

        kernel_reshape = kernel.reshape(out_channels, -1)
        grad_input_fold = kernel_reshape.transpose(0, 1).matmul(grad_out_reshaped)
        grad_input = col2im_indices(grad_input_fold, input.size(), kernel_size_y, kernel_size_x, padding=padding,
                                    stride=stride)

        return (grad_input, grad_kernel)


@register_function("batchnorm")
class AutogradBatchNorm(AutogradFunction):
    @staticmethod
    def forward(
            ctx,
            x,
            weight,
            bias,
            running_mean=None,
            running_var=None,
            inv_running_var=None,
            training=False,
            eps=1e-05,
            momentum=0.1,
    ):
        """
        Computes forward step of batch norm by normalizing x
            and returning weight * x_norm + bias.

        Running mean and var are computed over the `C` dimension for an input
        of size `(N, C, +)`.

        Note: inv_var can introduce precision errors due to sqrt and division
            particularly when the number of samples in a batch is small.

        Args:
            ctx (autograd_cyptensor.AutogradContext): context which
                stores parameters such as weight and bias for backward step.
            input (tuple of torch.tensors or cryptensor):
                containing (x, weight, bias) with shapes `(N, C, +)`, `C`, and `C`
                in turn.
            training (bool): if training is True, running mean and var are
                updated with the momentum factor and stored in module. Forward
                is performed using batch statistics. If training is False,
                running statistics are used and therefore cannot be none.
            running_mean (torch.tensor or cryptensor): with shape `C`
            running_var (torch.tensor or cryptensor): with shape `C`
            eps (float): specifies epsilon used for numerical precision in inv_var
            momentum (float): moment factor used in updating running mean and var.

        Returns: (weight * normalized input + bias) of shape `(N, C, +)`.
        """

        # determine dimensions over which means and variances are computed:
        stats_dimensions = list(range(x.dim()))
        stats_dimensions.pop(1)

        # shape for broadcasting statistics with input:
        broadcast_shape = [1] * x.dim()
        broadcast_shape[1] = x.shape[1]

        # compute mean and variance, track batch statistics:
        if training:
            raise NotImplementedError(
                "Batchnormalization is not currently supported for training"
            )
        else:
            if running_mean is None or running_var is None:
                raise ValueError(
                    "Must provide running_mean and running_var when training is False"
                )
            mean = running_mean
            variance = running_var

        # mean = torch.ones_like(running_mean.share.data, dtype=torch.float)
        # variance = torch.zeros_like(variance.share.data, dtype=torch.float)

        # compute inverse variance:
        # if torch.is_tensor(variance):
        #     inv_var = 1.0 / torch.sqrt(variance + eps)
        # else:
        #     inv_var = (variance + eps).pos_pow(-0.5)

        inv_var = inv_running_var

        # reshape shape (C) to broadcastable (1, C, 1, +):
        mean = mean.reshape(broadcast_shape)
        inv_var = inv_var.reshape(broadcast_shape)
        weight = weight.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)

        # compute z-scores:
        x_norm = (x - mean) * inv_var

        # save context and return:
        ctx.save_multiple_for_backward((x_norm, weight, inv_var, training))
        return x_norm * weight + bias

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient with respect to x, weight, and bias.

        Statistics are assumed to be computed along dimension C
        for an input of shape (N, C, ...). Note, partials with respect to
        the input treat mean and variance as constants similar to torch.

        Args:
            ctx (autograd_cyptensor.AutogradContext): context containing
                x_norm, weight, and inv_var. Note weight
                and inv_var must be broadcastable with grad_output.
            grad_output (cryptensor): batchnorm output of shape (N, C, +).

        Returns:
            x_grad (cryptensor): gradient with respect to x with shape (N, C, +).
            weight_grad (cryptensor): gradient with respect to the weight of
                with shape (C).
            bias_grad (cryptensor): gradient with respect to bias of shape (C).
        """

        # retrieve context:
        x_norm, weight, inv_var, training = ctx.saved_tensors

        # determine dimensions over which means and variances are computed:
        stats_dimensions = list(range(len(grad_output.shape)))
        stats_dimensions.pop(1)

        # shape for broadcasting statistics with output gradient:
        broadcast_shape = [1] * grad_output.dim()
        broadcast_shape[1] = grad_output.shape[1]

        # compute gradient w.r.t. weight:
        grad_weight = grad_output.mul(x_norm)
        grad_weight = grad_weight.sum(stats_dimensions)

        # compute gradient w.r.t. bias:
        grad_bias = grad_output.sum(stats_dimensions)

        # compute gradient with respect to the input:
        grad_output = grad_output.mul(weight)
        grad_input = grad_output.mul(inv_var)
        if training:
            # compute gradient term that is due to the mean:
            num_element = reduce(
                lambda x, y: x * y, [grad_output.size(d) for d in stats_dimensions]
            )
            grad_mean = grad_output.sum(stats_dimensions)
            grad_mean = grad_mean.reshape(broadcast_shape)
            grad_mean = grad_mean.mul(inv_var.div(-num_element))

            # compute gradient term that is due to the standard deviation:
            grad_std = x_norm.mul(grad_output).sum(stats_dimensions)
            grad_std = grad_std.reshape(broadcast_shape)
            grad_std = x_norm.mul(grad_std).mul(inv_var.div(-num_element))

            # put all the terms together:
            grad_input = grad_input.add(grad_mean).add(grad_std)

        # return gradients:
        return (grad_input, grad_weight, grad_bias)


@register_function("softmax")
class AutogradSoftmax(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        probs = input.softmax(dim)
        ctx.save_multiple_for_backward([probs, dim])
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        print('come softmax backward')
        probs, dim = ctx.saved_tensors
        if grad_output.dim() == 0 or grad_output.size(dim) == 1:
            return grad_output.new(torch.zeros(grad_output.size()))
        return grad_output.add(-probs.mul(grad_output).sum(dim, keepdim=True)).mul_(
            probs
        )


@register_function("log_softmax")
class AutogradLogSoftmax(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        probs = input.log_softmax(dim)
        ctx.save_multiple_for_backward([probs, dim])
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        print('come lod_softmax backward')
        probs, dim = ctx.saved_tensors
        if grad_output.dim() == 0 or grad_output.size(dim) == 1:
            return grad_output.new(torch.zeros(grad_output.size()))
        z = probs.exp()
        result = grad_output - z * grad_output.sum(dim, keepdim=True)
        return result


@register_function("cross_entropy")
class AutogradCrossEntropy(AutogradFunction):
    @staticmethod
    @timer_softmax
    def forward(ctx, pred, target, skip_forward=False):
        # NOTE: target is assumed to be one-hot vector.
        # Hard coded value for testing purpose
        # print('come cross_entropy')
        # print(f'size(pred)={pred.size()}')
        softmax = pred.softmax(1)
        ctx.save_multiple_for_backward([softmax, target])
        ctx.mark_non_differentiable(target)
        if skip_forward:
            return softmax.new(0)

        # print(f'softmax={softmax.size()}')
        # print(f'target={target.size()}')
        # Compute full forward pass
        loss_values = softmax.log().mul_(target).neg_()
        # print(f'loss_values={loss_values}')
        # print(f'loss_values={loss_values.size()}')
        out = loss_values.sum().div_(target.size(0))
        # print(f'outs={type(out)}')
        # print(f'out={out.get_plain_text(0)}')
        return out

    @staticmethod
    @timer_softmax
    def backward(ctx, grad_output):
        # print('come cross_entropy backward')
        # print(f'size(grad_output)={grad_output.size()}')
        # print(f'grad_output={grad_output.get_plain_text(0)}')
        softmax, target = ctx.saved_tensors
        loss_grad = softmax.sub(target)
        # print(f'size(loss_grad)={loss_grad.size()}')
        return loss_grad.div_(target.size(0)).mul_(grad_output)


""" 修改部分 """


# @register_function("relu")
# class AutogradReLU(AutogradFunction):
#     @staticmethod
#     @timer_relu
#     def forward(ctx, input):
#         from crypten.mpc import MPCTensor
#         from crypten.mpc.primitives import resharing
#         from crypten.mpc.primitives.converters import get_msb
#         from crypten import communicator as comm
#         # print("来gradients.py relu")
#         mask = get_msb(input._tensor)^1
#         ctx.save_for_backward(mask)
#         return resharing.mixed_mul(input._tensor, mask)


#     @staticmethod
#     @timer_relu
#     def backward(ctx, grad_output):
#         from crypten.mpc import MPCTensor
#         from crypten.mpc.primitives import resharing
#         from crypten import communicator as comm
#         (mask,) = ctx.saved_tensors

#         return resharing.mixed_mul(grad_output._tensor, mask)

@register_function("relu")
class AutogradReLU(AutogradFunction):
    @staticmethod
    @timer_rule
    def forward(ctx, input):
        from crypten.mpc.gw_relu_helper import gw_get_msb, mixed_mul_aby3, select_share
        import logging
        # print("来gradients.py falcon这")
        mask = gw_get_msb(input._tensor)^1
        ctx.save_for_backward(mask)
        out = mixed_mul_aby3(input._tensor, mask)
        comm.get().time_relu -= (comm.get().time_precomq)
        comm.get().time_precomq = 0
        return out

    @staticmethod
    @timer_rule
    def backward(ctx, grad_output):
        from crypten.mpc.gw_relu_helper import mixed_mul_aby3,select_share
        (mask,) = ctx.saved_tensors
        out = mixed_mul_aby3(grad_output._tensor, mask)
        comm.get().time_relu -= (comm.get().time_precomq)
        comm.get().time_precomq = 0
        return out


@register_function("max_pool2d")
class AutogradMaxPool2D(AutogradFunction):
    
    @staticmethod
    @timer_pool
    def forward(ctx, input, kernel_size, padding=0, stride=None, return_indices=False):
        # print('come maxpool')
        # 确定步长、内核大小、padding值均为长为2的tunple
        if stride is None:
            stride = kernel_size
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        if isinstance(kernel_size, (int, float)):
            kernel_size = (kernel_size, kernel_size)

        # perform max pooling:
        # Note return_indices is required to be True to computing backward.
        output, indices = input.max_pool2d(
            kernel_size, padding=padding, stride=stride, return_indices=True
        )

        # store information for backward pass and return:
        ctx.save_multiple_for_backward(
            (input.size(), indices, kernel_size, padding, stride)
        )
        comm.get().time_pool -= comm.get().time_precomq
        comm.get().time_precomq = 0
        if return_indices:
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            return output

    
    @staticmethod
    @timer_pool
    def backward(ctx, grad_output):
        output_size, indices, kernel_size, padding, stride = ctx.saved_tensors
        assert stride[0] == stride[1], "stride must be same in all axes"
        assert padding[0] == padding[1], "padding must be same in all axes"

        # 确保indices是普通的torch.tensor int64
        if not isinstance(indices, torch.Tensor):
            # print(type(indices))
            indices = indices.get_plain_text().to(dtype=torch.long, device=grad_output.device)

        # 保证是torch.tensor int64
        out = grad_output.share
        if isinstance(out, CUDALongTensor):
            out = out.tensor()

        # 只在indices对应的位置填充，其余为0
        assert indices.dim() == 4, 'max_pool2d backward only support dim=4,but it can Scalable to support dim >4'

        # 转换为两个doubleTensor来做
        block1, block2 = [(out >> (32 * i)) & (2 ** 32 - 1) for i in range(2)]
        block1, block2 = block1.double(), block2.double()

        max_unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride)

        # 满足调用UNmaxpoold的要求
        out1 = max_unpool(block1, indices, output_size=output_size)
        out1 = out1.long()
        out2 = max_unpool(block2, indices, output_size=output_size)
        out2 = out2.long()

        out = out1 + (out2 << 32)

        if out.device.type == 'cuda':
            out = CUDALongTensor(out)
        from crypten.mpc import MPCTensor
        out = MPCTensor.from_shares(out, src=comm.get().get_rank())

        comm.get().time_pool -= comm.get().time_precomq
        comm.get().time_precomq = 0

        return out