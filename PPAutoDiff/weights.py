import paddle
import torch
import numpy
import yaml
import os.path as osp
from .utils import map_for_each_weight, map_for_each_sublayer

def _assign_weight(paddle_sublayer, torch_submodule, param_name, paddle_param, torch_param):

    def _do_assign(param, np_value, type="paddle"): 
        assert list(param.shape) == list(np_value.shape), ("Shape is not the same. {} vs {} \n"
                    "Hint: \n"
                    "      1. check whether your paddle model definition and torch model definition are corresponding.\n"
                    "      2. check the weight shape of paddle:`{}` and torch:`{}` is the same.\n"
                    ).format(param.shape, np_value.shape, paddle_sublayer, torch_submodule)
        if type == "paddle":
            paddle.assign(paddle.to_tensor(np_value), param)
        elif type == "torch":
            param.data = torch.as_tensor(np_value).type(param.dtype)
        else: 
            raise RuntimeError("Invalid Arguments, type must be one of ['paddle', 'torch'].")

    shape = paddle_param.shape
    np_value = paddle.randn(shape).numpy()
    yaml_path = osp.join(osp.dirname(__file__), "configs", "assign_weight.yaml")
    assign_config = yaml.safe_load(open(yaml_path, "r"))
    config = assign_config.get(paddle_sublayer.__class__.__name__, None)
    if config is not None: 
        assert torch_submodule.__class__.__name__ == config['torch'], "Not correspond, check your __init__ to make sure every sublayer is corresponded."
    if config is None or param_name not in config['param']: 
        _do_assign(paddle_param, np_value, "paddle")
        _do_assign(torch_param, np_value, "torch")
    else: 
        """
        TODO: has more options? make it more elegant, remove the if-elif by MethodClass.
        """
        if config['param'][param_name] == "transpose": 
            _do_assign(paddle_param, np_value, "paddle")
            _do_assign(torch_param, numpy.transpose(np_value), "torch")


def assign_weight(layer, module):
    map_for_each_weight(_assign_weight, layer, module)


def check_weight_grad(layer, module, options):

    _weight_check = True
    _grad_check = True

    def _check_weight_grad(paddle_sublayer, torch_submodule, param_name, paddle_param, torch_param):
        nonlocal _weight_check, _grad_check
        atol = options.get("atol", 1e-7)
        transpose = False
        yaml_path = osp.join(osp.dirname(__file__), "configs", "assign_weight.yaml")
        assign_config = yaml.safe_load(open(yaml_path, "r"))
        config = assign_config.get(paddle_sublayer.__class__.__name__, None)
        if config is not None: 
            assert torch_submodule.__class__.__name__ == config['torch'], "Not correspond, check your __init__ to make sure every sublayer is corresponded."
            if param_name in config['param'] and config['param'][param_name] == "transpose": 
                transpose = True

        p_param = paddle_param.numpy()
        t_param = torch_param.detach().numpy()
        p_grad = paddle_param.grad.numpy()
        t_grad = torch_param.grad.detach().numpy()
        if transpose:
            t_param = numpy.transpose(t_param)
            t_grad = numpy.transpose(t_grad)

        assert list(p_param.shape) == list(t_param.shape), ("After training, the shape is different for param `{}`.\n"
                    "paddle: shape `{}` at `{}`.\n"
                    "torch: shape `{}` at `{}`.\n"
                    ).format(param_name, p_param.shape, paddle_sublayer, t_param.shape, torch_submodule)

        if not numpy.allclose(p_param, t_param, atol=atol):
            _weight_check = False
            print("After training, weight value is different for param `{}`.\n"
                    "paddle: at `{}`.\n"
                    "torch: at `{}`.\n"
                    .format(param_name, paddle_sublayer, torch_submodule))

        
        if not numpy.allclose(p_grad, t_grad, atol=atol):
            _grad_check = False 
            print("After training, grad value is different for param `{}`.\n"
                    "paddle: at `{}`.\n"
                    "torch: at `{}`.\n"
                    .format(param_name, paddle_sublayer, torch_submodule))
    
    map_for_each_weight(_check_weight_grad, layer, module)
    return _weight_check, _grad_check


def remove_inplace(layer, module):
    def _remove_inplace(layer, module):
        if hasattr(module, "inplace"): 
            module.inplace = False
    map_for_each_sublayer(_remove_inplace, layer, module)
    
