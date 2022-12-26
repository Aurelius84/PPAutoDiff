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
        self.relu = paddle.nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SimpleModule(torch.nn.Module): 
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)    
        self.linear2 = torch.nn.Linear(100, 10)    
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TestCaseName(unittest.TestCase):
    def test_success(self):
        layer = SimpleLayer()
        module = SimpleModule()
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        assert autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4}) == True, "Failed. expected success."

if __name__ == "__main__":
    unittest.main()

