import paddle
import torch
import numpy
import yaml
from functools import partial
from .report import Report, report_guard, print_report, current_report
import os.path as osp
import contextlib



def _assign_weight(paddle_sublayer, torch_submodule, param_name, paddle_param, torch_param, np_value):

    def _do_assign(param, np_value, type="paddle"): 
        assert list(param.shape) == list(np_value.shape), ("Shape is not the same. {} vs {} \n"
                    "Hint: \n"
                    "      1. check whether your paddle model definition and torch model definition are corresponding.\n"
                    "      2. check the weight shape of paddle:`{}` and torch:`{}` is the same.\n"
                    ).format(param.shape, np_value.shape, paddle_sublayer, torch_submodule)
        if type == "paddle":
            paddle.assign(np_value, param)
        elif type == "torch":
            param.data = torch.as_tensor(np_value).type(param.dtype)
        else: 
            raise RuntimeError("Invalid Arguments, type must be one of ['paddle', 'torch'].")

    yaml_path = osp.join(osp.dirname(__file__), "configs", "assign_weight.yaml")
    assign_config = yaml.safe_load(open(yaml_path, "r"))
    config = assign_config[paddle_sublayer.__class__.__name__]
    assert torch_submodule.__class__.__name__ == config['torch'], "Not correspond, check your __init__ to make sure every sublayer is corresponded."
    if param_name not in config['param']: 
        _do_assign(paddle_param, np_value, "paddle")
        _do_assign(torch_param, np_value, "torch")
    else: 
        """
        TODO: has more options? make it more elegant, remove the if-elif by MethodClass.
        """
        if config['param'][param_name] == "transpose": 
            _do_assign(paddle_param, np_value, "paddle")
            _do_assign(torch_param, numpy.transpose(np_value), "torch")


def _auto_fill_weight(layer, module): 
    """
    Automatically fill weights by randn.
    """
    for paddle_sublayer, torch_submodule in zip(layer.sublayers(True), module.modules()): 
        for (name, paddle_param), torch_param in zip(paddle_sublayer.named_parameters("",False), torch_submodule.parameters(False)): 
            shape = paddle_param.shape
            rand_value = paddle.randn(shape).numpy()
            _assign_weight(
                 paddle_sublayer, torch_submodule,
                 name, paddle_param, torch_param, rand_value)


def autodiff(layer, module, example_inp, auto_weights=True, options={}): 
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer): 
        module (torch.nn.Module): 
        example_inp (numpy.array):
        auto_weights (boolean, optional):
        options (dict, optional):
    Returns:
        paddle_output, torch_output
    """
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    assert isinstance(module, torch.nn.Module), "Invalid Argument."
    assert isinstance(example_inp, numpy.ndarray), "Invalid Argument."

    paddle.set_device('cpu')
    module = module.cpu()

    if auto_weights: 
        _auto_fill_weight(layer, module)

    torch_report = Report("torch")
    paddle_report = Report("paddle")
    with report_guard(torch_report): 
        with _register_torch_hooker(module): 
            try: 
                torch_output = module(torch.as_tensor(example_inp))
            except Exception as e: 
                raise RuntimeError("Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(str(e)))

    with report_guard(paddle_report): 
        with _register_paddle_hooker(layer):
            try: 
                paddle_output = layer(paddle.to_tensor(example_inp))
            except Exception as e: 
                raise RuntimeError("Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(str(e)))

    print_report(torch_report, paddle_report, options)
    return paddle_output, torch_output


@contextlib.contextmanager
def _register_paddle_hooker(layer):
    def hook(module, input, output, idx):
        rep = current_report()
        rep.put_item(input, output, module, idx)
        return None

    remove_handles = []
    for idx, mod in enumerate(layer.sublayers(True)): 
        handle = mod.register_forward_post_hook(partial(hook, idx=idx))
        if remove_handles: 
            remove_handles.append(handle)
    yield
    for h in remove_handles: 
        h.remove()

    
@contextlib.contextmanager
def _register_torch_hooker(module):
    def hook(module, input, output, idx):
        rep = current_report()
        rep.put_item(input, output, module, idx)
        return None

    remove_handles = []
    for idx, mod in enumerate(module.modules()): 
        handle = mod.register_forward_hook(partial(hook, idx=idx))
        remove_handles.append(handle)
    yield
    for h in remove_handles: 
        h.remove()
