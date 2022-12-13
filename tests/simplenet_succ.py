from PPAutoDiff import autodiff
import torch
import paddle
import numpy

"""
torch 和 paddle 的weight的格式不同，有转置的区别，所以我们使用对称矩阵。
"""
w1 = numpy.random.randn(100, 100).astype("float32") 
w2 = numpy.random.randn(100, 100).astype("float32")
w1 = w1 + numpy.transpose(w1)
w2 = w2 + numpy.transpose(w2)


class SimpleLayer(paddle.nn.Layer): 
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100, bias_attr=False)    
        self.linear2 = paddle.nn.Linear(100, 100, bias_attr=False)    
        paddle.assign(paddle.to_tensor(w1), self.linear1.weight)
        paddle.assign(paddle.to_tensor(w2), self.linear2.weight)
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
        self.linear1 = torch.nn.Linear(100, 100, bias=False)    
        self.linear2 = torch.nn.Linear(100, 100, bias=False)    
        self.linear1.weight.data = torch.as_tensor(w1)
        self.linear2.weight.data = torch.as_tensor(w2)
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

def main():
    layer = SimpleLayer()
    module = SimpleModule()
    inp = paddle.rand((8, 100)).numpy().astype("float32")
    #module(torch.as_tensor(inp))
    autodiff(layer, module, inp, {'atol': 1e-3})  # SUCCESS
    #autodiff(layer, module, inp, {'atol': 1e-5}) # FAILED
    
if __name__ == "__main__":
    main()
