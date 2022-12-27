from PPAutoDiff import autodiff
from PPAutoDiff.weights import check_weight_grad
import torch
import paddle
import numpy
import unittest

class SimpleLayer(paddle.nn.Layer): 
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)    
        self.linear2 = paddle.nn.Linear(100, 10)    
        self.act = paddle.nn.ReLU()
        
    def forward(self, x):
        """
        x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
        |                                  |
        |----------------------------------|
        """
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual 
        x = self.linear2(x)
        return x

class SimpleModule(torch.nn.Module): 
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)    
        self.linear2 = torch.nn.Linear(100, 10)    
        self.act = torch.nn.ReLU()
        
    def forward(self, x):
        """
        x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
        |                                  |
        |----------------------------------|
        """
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual 
        x = self.linear2(x)
        return x


class TestCaseName(unittest.TestCase):
    def test_check_weight_grad(self):
        layer = SimpleLayer()
        module = SimpleModule()

        inp = paddle.rand((100, 100)).numpy().astype("float32")
        assert autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4}) == True, "Failed. expected success."
        
        module.zero_grad()
        weight_check, grad_check = check_weight_grad(layer, module, options={'atol': 1e-4})
        assert weight_check == True, "Weight params should be same"
        assert grad_check == False, "Grad should be different"

        layer = SimpleLayer()
        module = SimpleModule()
        assert autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4}) == True, "Failed. expected success."
        
        for param in module.parameters():
            param.data = param * 2
        weight_check, grad_check = check_weight_grad(layer, module, options={'atol': 1e-4})
        assert weight_check == False, "Weight params should be different"
        assert grad_check == True, "Grad should be same"

if __name__ == "__main__":
    unittest.main()
