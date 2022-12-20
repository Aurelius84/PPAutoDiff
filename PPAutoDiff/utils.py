import paddle
import torch
from paddle.fluid.layers.utils import flatten, to_sequence, map_structure, pack_sequence_as
import contextlib

def is_tensor(x):
    return isinstance(x, (paddle.Tensor, torch.Tensor))

def is_tensors(*x):
    ret = True
    for i in x: 
       ret = ret and is_tensor(i) 
    return ret

def is_require_grad(x):
    if hasattr(x, "requires_grad"): 
        return x.requires_grad
    if hasattr(x, "stop_gradient"): 
        return not x.stop_gradient
    return False

def for_each_tensor(*structure):
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    entries = filter(lambda x: is_tensors(*x), entries)
    for tensors in entries: 
        yield tensors

def for_each_grad_tensor(*structure):
    def filter_fn(ts):
        return is_tensors(*ts) and is_require_grad(ts[0])
    for ts in filter(filter_fn, for_each_tensor(*structure)):
        yield ts
