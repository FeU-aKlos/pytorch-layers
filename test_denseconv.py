import unittest

from denseconv import _bn_function_factory,DenseLayer,DenseBlock,Transition

class TestDenseLayer(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()