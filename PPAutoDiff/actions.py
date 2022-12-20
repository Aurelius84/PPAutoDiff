import numpy as np
from .utils import is_tensor

class ActionPool: 
    def __init__(self):
        self.pool = []

    def register(self, cls): 
        name = cls.__name__
        self.pool.append(cls())
        sorted(self.pool, key=lambda x: x.priority, reverse=True)
        return cls

    def find_actions(self, torch_net, paddle_net):
        for act in self.pool: 
            if act.match(torch_net, paddle_net):
                return act
        raise RuntimeError("No action is matched, not expected.")

global_actions = ActionPool()

def get_action(*args, **kargs):
    return global_actions.find_actions(*args, **kargs)

class Action:
    def match(self, torch_net, paddle_net): 
        raise NotImplementedError("")

    def __call__(self, torch_item, paddle_item, cfg):
        raise NotImplementedError("")

    @property
    def priority(self):
        raise NotImplementedError("")


@global_actions.register
class EqualAction(Action):
    def match(self, torch_net, paddle_net): 
        return True

    @property
    def priority(self):
        return 0

    def __call__(self, torch_item, paddle_item, cfg):
        """
        NOTE:
        """
        atol = cfg.get("atol", 1e-7)
        torch_tensors = torch_item.compare_tensors()
        paddle_tensors = paddle_item.compare_tensors()
        for (tt,), (pt,) in zip(torch_tensors, paddle_tensors): 
            np.testing.assert_allclose(
                tt.detach().numpy(), 
                pt.numpy(), 
                atol=atol)

