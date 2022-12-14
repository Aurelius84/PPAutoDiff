from PPAutoDiff import autodiff
import torch
import paddle
import numpy
import unittest
import torchvision

class TestCaseName(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_success(self):
        layer = paddle.vision.resnet50()
        module = torchvision.models.resnet50()
        inp = paddle.rand((10, 3, 224, 224)).numpy().astype("float32")
        assert autodiff(layer, module, inp, auto_weights=True, options={'atol': 5e-2}) == True, "Failed. expected success."

if __name__ == "__main__":
    unittest.main()
