from utils import ImageLoader
import unittest


class TestImageLoader(unittest.TestCase):

    def test_pbm(self):
        img = ImageLoader.load("./tests/images/pbm_test.pbm")
        self.assertEqual(img.shape, (7, 24))

    def test_pgm(self):
        img = ImageLoader.load("./tests/images/pgm_test.pgm")
        self.assertEqual(img.shape, (480, 640))
    
    def test_ppm(self):
        img = ImageLoader.load("./tests/images/ppm_test.ppm")
        self.assertEqual(img.shape, (281, 500, 3))


if __name__ == '__main__':
    unittest.main()