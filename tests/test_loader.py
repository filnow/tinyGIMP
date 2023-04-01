import os,sys
import unittest
sys.path.append(os.path.abspath(os.path.join('.')))

from utils import ImageLoader


class TestImageLoader(unittest.TestCase):
    #ASCII from tests/images
    def test_pbm(self):
        img = ImageLoader.load("./tests/images/pbm_test.pbm")
        self.assertEqual(img.shape, (7, 24))

    def test_pgm(self):
        img = ImageLoader.load("./tests/images/pgm_test.pgm")
        self.assertEqual(img.shape, (480, 640))
    
    def test_ppm(self):
        img = ImageLoader.load("./tests/images/ppm_test.ppm")
        self.assertEqual(img.shape, (281, 500, 3))
    #ASCII from data
    def test_pbm2(self):
        img = ImageLoader.load("./data/test_ascii.pbm")
        self.assertEqual(img.shape, (100, 100))

    def test_pgm2(self):
        img = ImageLoader.load("./data/test_ascii.pgm")
        self.assertEqual(img.shape, (100, 100))
    
    def test_ppm2(self):
        img = ImageLoader.load("./data/test_ascii.ppm")
        self.assertEqual(img.shape, (100, 100, 3))
    #RAW from data
    def test_pbm3(self):
        img = ImageLoader.load("./data/test_raw.pbm")
        self.assertEqual(img.shape, (100, 200, 3))
    
    def test_pgm3(self):
        img = ImageLoader.load("./data/test_raw.pgm")
        self.assertEqual(img.shape, (100, 100, 3))
    
    def test_ppm3(self):
        img = ImageLoader.load("./data/test_raw.ppm")
        self.assertEqual(img.shape, (100, 100, 3))


if __name__ == '__main__':
    unittest.main()