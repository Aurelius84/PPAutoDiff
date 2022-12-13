from PPAutoDiff import autodiff
import torch
import paddle
import numpy


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

def main():
    layer = SimpleLayer()
    module = SimpleModule()
    inp = paddle.rand((8, 100)).numpy().astype("float32")
    #module(torch.as_tensor(inp))
    autodiff(layer, module, inp)
    
if __name__ == "__main__":
    main()
