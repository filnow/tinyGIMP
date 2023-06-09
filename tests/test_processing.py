import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from utils import ImageProcessor, ImageLoader

class TestImageProcessor_ppm(unittest.TestCase):
	def __init__(self, methodName: str = "runTest") -> None:
		super().__init__(methodName)
		self.img_ppm = ImageLoader.load("./tests/images/ppm_test.ppm")
    
	def test_desaturate(self):
		img = ImageProcessor(self.img_ppm).desaturate()
		self.assertEqual(img.shape, (281, 500))

	def test_negative(self):
		img = ImageProcessor(self.img_ppm).negative()
		self.assertEqual(img.shape, (281, 500, 3))

	def test_contrast_linear(self):
		img = ImageProcessor(self.img_ppm).contrast(1.5, "linear")
		self.assertEqual(img.shape, (281, 500, 3))

	def test_contrast_log(self):
		img = ImageProcessor(self.img_ppm).contrast(1.5, "log")
		self.assertEqual(img.shape, (281, 500, 3))

	def test_contrast_power(self):
		img = ImageProcessor(self.img_ppm).contrast(1.5, "power")
		self.assertEqual(img.shape, (281, 500, 3))

	def test_brightness(self):
		img = ImageProcessor(self.img_ppm).brightness(2)
		self.assertEqual(img.shape, (281, 500, 3))

	def test_saturation(self):
		img = ImageProcessor(self.img_ppm).saturation(2)
		self.assertEqual(img.shape, (281, 500, 3))
	
	def test_calculations_sum(self):
		img = ImageProcessor(self.img_ppm).calculations(self.img_ppm, "sum")
		self.assertEqual(img.shape, (281, 500, 3))
	
	def test_calculations_sub(self):
		img = ImageProcessor(self.img_ppm).calculations(self.img_ppm, "subtraction")
		self.assertEqual(img.shape, (281, 500, 3))
	
	def test_calculations_mul(self):
		img = ImageProcessor(self.img_ppm).calculations(self.img_ppm, "multiplication")
		self.assertEqual(img.shape, (281, 500, 3))


class TestImageProcessor_pbm(unittest.TestCase):
	def __init__(self, methodName: str = "runTest") -> None:
		super().__init__(methodName)
		self.img_pbm = ImageLoader.load("./tests/images/pbm_test.pbm")

	def test_negative(self):
		img = ImageProcessor(self.img_pbm).negative()
		self.assertEqual(img.shape, (7, 24))

	def test_contrast_linear(self):
		img = ImageProcessor(self.img_pbm).contrast(1.5, "linear")
		self.assertEqual(img.shape, (7, 24))

	def test_contrast_log(self):
		img = ImageProcessor(self.img_pbm).contrast(1.5, "log")
		self.assertEqual(img.shape, (7, 24))

	def test_contrast_power(self):
		img = ImageProcessor(self.img_pbm).contrast(1.5, "power")
		self.assertEqual(img.shape, (7, 24))

	def test_brightness(self):
		img = ImageProcessor(self.img_pbm).brightness(2)
		self.assertEqual(img.shape, (7, 24))
	
	def test_calculations_sum(self):
		img = ImageProcessor(self.img_pbm).calculations(self.img_pbm, "sum")
		self.assertEqual(img.shape, (7, 24))
	
	def test_calculations_sub(self):
		img = ImageProcessor(self.img_pbm).calculations(self.img_pbm, "subtraction")
		self.assertEqual(img.shape, (7, 24))
	
	def test_calculations_mul(self):
		img = ImageProcessor(self.img_pbm).calculations(self.img_pbm, "multiplication")
		self.assertEqual(img.shape, (7, 24))


class TestImageProcessor_pgm(unittest.TestCase):
	def __init__(self, methodName: str = "runTest") -> None:
		super().__init__(methodName)
		self.img_pgm = ImageLoader.load("./tests/images/pgm_test.pgm")

	def test_negative(self):
		img = ImageProcessor(self.img_pgm).negative()
		self.assertEqual(img.shape, (480, 640))

	def test_contrast_linear(self):
		img = ImageProcessor(self.img_pgm).contrast(1.5, "linear")
		self.assertEqual(img.shape, (480, 640))

	def test_contrast_log(self):
		img = ImageProcessor(self.img_pgm).contrast(1.5, "log")
		self.assertEqual(img.shape, (480, 640))

	def test_contrast_power(self):
		img = ImageProcessor(self.img_pgm).contrast(1.5, "power")
		self.assertEqual(img.shape, (480, 640))

	def test_brightness(self):
		img = ImageProcessor(self.img_pgm).brightness(2)
		self.assertEqual(img.shape, (480, 640))
	
	def test_calculations_sum(self):
		img = ImageProcessor(self.img_pgm).calculations(self.img_pgm, "sum")
		self.assertEqual(img.shape, (480, 640))
	
	def test_calculations_sub(self):
		img = ImageProcessor(self.img_pgm).calculations(self.img_pgm, "subtraction")
		self.assertEqual(img.shape, (480, 640))
	
	def test_calculations_mul(self):
		img = ImageProcessor(self.img_pgm).calculations(self.img_pgm, "multiplication")
		self.assertEqual(img.shape, (480, 640))


if __name__ == '__main__':
    unittest.main()