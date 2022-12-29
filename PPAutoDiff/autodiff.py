import paddle
import torch
import numpy
from functools import partial
from .report import Report, report_guard, check_forward_and_backward, current_report
import contextlib
from paddle.fluid.layers.utils import flatten, to_sequence, map_structure, pack_sequence_as
from .weights import assign_weight, remove_inplace, check_weight_grad
from .utils import for_each_grad_tensor
from .stack_info import *
import traceback

def autodiff(layer, module, example_inp, auto_weights=True, options={}): 
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer): 
        module (torch.nn.Module): 
        example_inp (numpy.array):
        auto_weights (boolean, optional):
        options (dict, optional):
            atol
    Returns:
        True for success, False for failed.
    """
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    assert isinstance(module, torch.nn.Module), "Invalid Argument."
    assert isinstance(example_inp, numpy.ndarray), "Invalid Argument."

    paddle.set_device("cpu")
    module = module.to("cpu")

    _preprocess(layer, module, example_inp, auto_weights, options)

    torch_report = Report("torch")
    paddle_report = Report("paddle")
    with report_guard(torch_report): 
        with _register_torch_hooker(module): 
            try: 
                torch_input = torch.as_tensor(example_inp)
                torch_input.requires_grad=True
                torch_output = module(torch_input)
                loss = torch_output.mean()
                loss.backward()
            except Exception as e: 
                raise RuntimeError("Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(str(e)))

    with report_guard(paddle_report): 
        with _register_paddle_hooker(layer):
            try: 
                paddle_input = paddle.to_tensor(example_inp)
                paddle_input.stop_gradient=False
                paddle_output = layer(paddle_input)
                loss = paddle_output.mean()
                loss.backward()
            except Exception as e: 
                raise RuntimeError("Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(str(e)))
    
    print ("Max output diff is {}\n".format(numpy.abs(paddle_output.numpy() - torch_output.detach().numpy()).max()))

    weight_check, grad_check = check_weight_grad(layer, module, options)
    ret = check_forward_and_backward(torch_report, paddle_report, options)
    ret = ret and weight_check and grad_check
    return ret 

def tensor_hook(x_grad, bwd_item, nth_tensor):
    #print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    bwd_item.set_input_grads(nth_tensor, x_grad)
    return x_grad
    
def layer_hook(module, input, output, idx):
    rep = current_report()
    frame_info, frames = extract_frame_summary()
    fwd_item = rep.put_item('forward', input, output, module, idx, frame_info, frames)
    bwd_item = rep.put_item('backward', input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)): 
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))
    return None

@contextlib.contextmanager
def _register_paddle_hooker(layer):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    for idx, mod in enumerate(layer.sublayers(True)): 
        handle = mod.register_forward_post_hook(partial(layer_hook, idx=idx))
        remove_handles.append(handle)
    yield
    for h in remove_handles: 
        h.remove()

    
@contextlib.contextmanager
def _register_torch_hooker(module):
    remove_handles = []
    for idx, mod in enumerate(module.modules()): 
        handle = mod.register_forward_hook(partial(layer_hook, idx=idx))
        remove_handles.append(handle)
    yield
    for h in remove_handles: 
        h.remove()

def _preprocess(layer, module, example_inp, auto_weights, options):
    remove_inplace(layer, module)
    if auto_weights: 
        assign_weight(layer, module)
