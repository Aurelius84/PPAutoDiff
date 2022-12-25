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


def check_weight_grad(layer, module):
    def _check_weight_grad(paddle_sublayer, torch_submodule, param_name, paddle_param, torch_param): 
        # TODO(zhanfei): add weigth compare function.
        pass
    map_for_each_weight(_check_weight_grad, layer, module)

def assign_weight(layer, module):
    map_for_each_weight(_assign_weight, layer, module)

def remove_inplace(layer, module):
    def _remove_inplace(layer, module):
        if hasattr(module, "inplace"): 
            module.inplace = False
    map_for_each_sublayer(_remove_inplace, layer, module)
    
