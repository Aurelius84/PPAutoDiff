from PPAutoDiff import autodiff
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
    def test_success(self):
        layer = SimpleLayer()
        module = SimpleModule()
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        assert autodiff(layer, module, inp, auto_weights=True) == True, "Failed. expected success."


    def test_failed(self):
        layer = SimpleLayer()
        module = SimpleModule()
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        assert autodiff(layer, module, inp, auto_weights=False) == False, "Success. expected failed."
        
if __name__ == "__main__":
    unittest.main()
