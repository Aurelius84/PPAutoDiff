import paddle
import torch
import numpy
from .report import Report, report_guard, print_report, current_report

def autodiff(layer, module, example_inp, options={}): 
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer): 
        module (torch.nn.Module): 
        example_inp (numpy.array):
        options (dict, optional):
    Returns:
        None

    """
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    assert isinstance(module, torch.nn.Module), "Invalid Argument."
    assert isinstance(example_inp, numpy.ndarray), "Invalid Argument."

    paddle.set_device('cpu')
    module = module.cpu()

    torch_report = Report("torch")
    paddle_report = Report("paddle")
    with report_guard(torch_report): 
        _register_torch_hooker(module)
        try: 
            torch_output = module(torch.as_tensor(example_inp))
        except Exception as e: 
            raise RuntimeError("Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(str(e)))

    with report_guard(paddle_report): 
        _register_paddle_hooker(layer)
        try: 
            paddle_output = layer(paddle.to_tensor(example_inp))
        except Exception as e: 
            raise RuntimeError("Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(str(e)))

    print_report(torch_report, paddle_report, options)


def _register_paddle_hooker(layer):
    # paddle and torch have the static polymorphism, share the same code. may be differ later.
    def hook(module, input, output):
        rep = current_report()
        rep.put_item(input, output, module)
        return None

    for mod in layer.sublayers(True): 
        mod.register_forward_post_hook(hook)
    

def _register_torch_hooker(module):
    def hook(module, input, output):
        rep = current_report()
        rep.put_item(input, output, module)
        return None

    for mod in module.modules(): 
        mod.register_forward_hook(hook)

