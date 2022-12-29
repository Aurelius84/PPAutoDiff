# PPAutoDiff
**P**addle and **P**ytorch model **Auto**matically **Diff** precision tools.

## 

## 使用说明

-   autodiff使用接口与参数说明

    接口函数签名：`autodiff(layer, module, example_inp, auto_weights=True, options={})`

    -   layer：传入paddle模型
    -   module：传入torch模型
    -   inp：传入输入数据（第一个维度是batch）
    -   auto_weights: 是否使用随机数值统一初始化paddle与torch模型，默认为True
    -   options：一个传递参数的字典，目前支持在字典中传入 'atol' 参数

-   使用注意点与样例代码：

    -   在使用autodiff时，需要传入paddle模型与torch模型，在模型定义时，需要将forward中所使用的子模型在__init__中定义，并保证其中的子模型定义顺序一致，具体可见下方示例代码

```py
from PPAutoDiff import autodiff
import torch
import paddle

# 使用paddle与torch定义相同结构的模型: SimpleLayer 和 SimpleModule
# 样例模型结构为:
#       x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
#       |                                  |
#       |----------------------------------|

# 注意：以下代码中linear1均定义在linear2之前，若linear1与linear2定义的顺序不对应，则将导致错误
class SimpleLayer(paddle.nn.Layer): 
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)    
        self.linear2 = paddle.nn.Linear(100, 10)    
        self.act = paddle.nn.ReLU()
        
    def forward(self, x):
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
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual 
        x = self.linear2(x)
        return x


layer = SimpleLayer()
module = SimpleModule()
inp = paddle.rand((100, 100)).numpy().astype("float32")
autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4})
```

-   autodiff的输出信息：

    -   当正确对齐时，autodiff将输出paddle与torch模型输出结果之间的最大diff值

        ```
        Max output diff is 6.866455078125e-05
        forward 4 steps compared.
        bacward 4 steps compared.
        SUCCESS !!!
        ```

    -   当模型对齐失败时，将输出：

        -   训练后，模型权重以及梯度出现diff的位置
        -   在训练过程中首先出现diff的位置（在forward过程或backward过程）
        -   paddle与torch的调用栈

        ```
        Max output diff is 4.716042518615723
        After training, weight value is different for param `weight`.
        paddle: at `Linear(in_features=100, out_features=100, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=100, bias=True)`.
        
        After training, grad value is different for param `weight`.
        paddle: at `Linear(in_features=100, out_features=100, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=100, bias=True)`.
        
        After training, weight value is different for param `bias`.
        paddle: at `Linear(in_features=100, out_features=100, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=100, bias=True)`.
        
        After training, grad value is different for param `bias`.
        paddle: at `Linear(in_features=100, out_features=100, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=100, bias=True)`.
        
        After training, weight value is different for param `weight`.
        paddle: at `Linear(in_features=100, out_features=10, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=10, bias=True)`.
        
        After training, grad value is different for param `weight`.
        paddle: at `Linear(in_features=100, out_features=10, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=10, bias=True)`.
        
        After training, weight value is different for param `bias`.
        paddle: at `Linear(in_features=100, out_features=10, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=10, bias=True)`.
        
        FAILED !!!
            Diff found in `Forward  Stagy` in step: 0, net_id is 1 vs 1
            Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>
        
        Not equal to tolerance rtol=1e-07, atol=0.0001
        
        Mismatched elements: 9997 / 10000 (100%)
        Max absolute difference: 2.43094
        Max relative difference: 39929.07
         x: array([[ 0.461714, -0.472631,  0.087497, ...,  0.253966, -0.018891,
                 0.155347],
               [ 0.350096, -0.104775,  0.25634 , ...,  0.215376,  0.028362,...
         y: array([[-0.551864, -0.335479,  0.591859, ...,  0.145943,  1.211705,
                -0.144882],
               [-0.526738,  0.505224,  0.178354, ..., -0.125375,  1.302669,...
        
        
        Paddle Stacks:
        =========================
                 File /workspace/PPAutoDiff/PPAutoDiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1005    _dygraph_call_func
                        hook_result = forward_post_hook(self, inputs, outputs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File test_simplenet1.py: 19    forward
                        x = self.linear1(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1002    _dygraph_call_func
                        outputs = self.forward(*inputs, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 54    autodiff
                        paddle_output = layer(paddle_input)
                 File test_simplenet1.py: 49    <module>
                        autodiff(layer, module, inp, auto_weights=False, options={'atol': 1e-4})
        Torch  Stacks:
        =========================
                 File /workspace/PPAutoDiff/PPAutoDiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                        hook_result = hook(self, input, result)
                 File test_simplenet1.py: 39    forward
                        x = self.linear1(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1148    _call_impl
                        result = forward_call(*input, **kwargs)
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 43    autodiff
                        torch_output = module(torch_input)
                 File test_simplenet1.py: 49    <module>
                        autodiff(layer, module, inp, auto_weights=False, options={'atol': 1e-4})
        ```

        以下是backward过程中出现diff时的报错信息

        ```
        Max output diff is 1.621246337890625e-05
        After training, grad value is different for param `weight`.
        paddle: at `Linear(in_features=100, out_features=100, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=100, bias=True)`.
        
        After training, grad value is different for param `bias`.
        paddle: at `Linear(in_features=100, out_features=100, dtype=float32)`.
        torch: at `Linear(in_features=100, out_features=100, bias=True)`.
        
        /workspace/PPAutoDiff/PPAutoDiff/utils.py:110: UserWarning: Warning: duplicate key is found, use list + pop strategy.
          warnings.warn("Warning: duplicate key is found, use list + pop strategy.")
        forward 4 steps compared.
        FAILED !!!
            Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2
            Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>
        
        Not equal to tolerance rtol=1e-07, atol=0.0001
        
        Mismatched elements: 9300 / 10000 (93%)
        Max absolute difference: 0.00227769
        Max relative difference: 53.632496
         x: array([[ 0.003637,  0.002246,  0.003323, ...,  0.006409,  0.001685,
                -0.001022],
               [ 0.003637,  0.002246,  0.003323, ...,  0.006409,  0.001685,...
         y: array([[ 0.004051,  0.00277 ,  0.004514, ...,  0.007816,  0.00193 ,
                -0.001609],
               [ 0.004051,  0.00277 ,  0.004514, ...,  0.007816,  0.00193 ,...
        
        
        Paddle Stacks:
        =========================
                 File /workspace/PPAutoDiff/PPAutoDiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1005    _dygraph_call_func
                        hook_result = forward_post_hook(self, inputs, outputs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File test_check_weight_grad.py: 36    forward
                        x3 = self.linear2(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1002    _dygraph_call_func
                        outputs = self.forward(*inputs, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 54    autodiff
                        paddle_output = layer(paddle_input)
                 File test_check_weight_grad.py: 57    <module>
                        autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4})
        Torch  Stacks:
        =========================
                 File /workspace/PPAutoDiff/PPAutoDiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                        hook_result = hook(self, input, result)
                 File test_check_weight_grad.py: 49    forward
                        x3 = self.linear2(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1148    _call_impl
                        result = forward_call(*input, **kwargs)
                 File /workspace/PPAutoDiff/PPAutoDiff/autodiff.py: 43    autodiff
                        torch_output = module(torch_input)
                 File test_check_weight_grad.py: 57    <module>
                        autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4})
        ```

        


