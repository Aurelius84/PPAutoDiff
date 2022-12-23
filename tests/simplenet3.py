from PPAutoDiff import autodiff
import torch
import paddle
import numpy
import unittest

"""
测试 同一个Module / Layer被多次forward

期待结果：
Success
"""

class SimpleLayer(paddle.nn.Layer): 
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)    
        self.linear2 = paddle.nn.Linear(100, 100)    
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear1(x)
        x3 = self.linear2(x)
        return x1 + x2 + x3

class SimpleModule(torch.nn.Module): 
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)    
        self.linear2 = torch.nn.Linear(100, 100)    
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear1(x)
        x3 = self.linear2(x)
        return x2 + x1 + x3

class TestCase(unittest.TestCase):
    def test_success(self):
        layer = SimpleLayer()
        module = SimpleModule()
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        assert autodiff(layer, module, inp, auto_weights=True) == True, "Failed, expect success."

if __name__ == "__main__":
    unittest.main()
