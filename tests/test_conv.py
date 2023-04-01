import os,sys
import unittest
sys.path.append(os.path.abspath(os.path.join('.')))

from utils import ImageLoader, Conv

class TestConv(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.img = ImageLoader.load("./data/cat.jpg")

    def test_conv(self):
        filtr = Conv([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
        uniform = filtr.uniform_blur(self.img, 3)
        self.assertAlmostEqual(uniform, filtr.apply(self.img))
    
if __name__ == "__main__":
    unittest.main()