import paddle
import torch
import numpy
import yaml
import os.path as osp

def map_for_each_weight(fn, layer, module): 
    """
    Automatically fill weights by randn.
    """
    for paddle_sublayer, torch_submodule in zip(layer.sublayers(True), module.modules()): 
        for (name, paddle_param), torch_param in zip(paddle_sublayer.named_parameters("",False), torch_submodule.parameters(False)): 
            fn(paddle_sublayer, torch_submodule, name, paddle_param, torch_param)


def _assign_weight(paddle_sublayer, torch_submodule, param_name, paddle_param, torch_param):

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

    shape = paddle_param.shape
    np_value = paddle.randn(shape).numpy()
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


def _check_weight_grad(paddle_sublayer, torch_submodule, param_name, paddle_param, torch_param): 
    # TODO: add weigth compare function.
    pass

