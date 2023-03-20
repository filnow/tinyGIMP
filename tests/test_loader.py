from utils import ImageLoader
import unittest


class TestImageLoader(unittest.TestCase):

    def test_pbm(self):
        img = ImageLoader.load("./data/pbm_test.pbm")
        self.assertEqual(img.shape, (24, 7))

    def test_pgm(self):
        img = ImageLoader.load("./data/pgm_test.pgm")
        self.assertEqual(img.shape, (24, 7))

if __name__ == '__main__':
    unittest.main()