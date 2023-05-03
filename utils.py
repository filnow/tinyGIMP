from typing import List
import cv2
import numpy as np


class ImageLoader:
    # NOTE: We use static methods because we don't need to instantiate the class
    @staticmethod
    def load(path: str) -> np.ndarray:
        image_type = ""
        with open(path, encoding='latin') as f:
            lines = [x for x in f.read().split()]
        for idx, line in enumerate(lines):
            if not line.isdigit():
                if line in ['P1', 'P2', 'P3']:
                    image_type = line
                continue
            else:
                break
        size = [int(lines[idx + 1]), int(lines[idx])] if image_type != "" else []
        image = ImageLoader._get_image(lines, size, idx, path, image_type)
        return image

    @staticmethod
    def _get_image(lines: List[str],
                   size: List[int],
                   idx: int,
                   path: str,
                   image_type: str) -> np.ndarray:
        if image_type == 'P1':
            image = np.asarray([(int(x) - 1) * -255 for row in lines[idx + 2:] for x in row if row.isdigit()]).reshape(
                size)
        elif image_type == 'P2':
            image = np.asarray([int(x) for x in lines[idx + 3:] if x.isdigit()]).reshape(size)
        elif image_type == 'P3':
            image = np.asarray([int(x) for x in lines[idx + 3:] if x.isdigit()]).reshape(*size, 3)
            # NOTE: Swap the red and blue channels
            image[:, :, [0, 2]] = image[:, :, [2, 0]]
        else:
            image = cv2.imread(path)
        return image


class ImageProcessor:
    def __init__(self, img: np.ndarray) -> None:
        self.img = img

    def desaturate(self) -> np.ndarray:
        return cv2.cvtColor(self.img.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.uint8)

    def negative(self) -> np.ndarray:
        lut: np.ndarray = np.arange(256)[::-1]
        return np.take(lut, self.img).astype(np.uint8)

    def contrast(self, factor: float, function: str) -> np.ndarray:
        if function == "linear":
            lut: np.ndarray = np.clip(np.arange(256) * factor, 0, 255)
            return np.take(lut, self.img).astype(np.uint8)
        elif function == "log":
            lut: np.ndarray = np.clip(255 * np.log(1 + np.arange(256)) / np.log(1 + 255 / factor), 0, 255)
            return np.take(lut, self.img).astype(np.uint8)
        elif function == "power":
            lut: np.ndarray = np.clip(255 * np.power(np.arange(256) / 255, 1 / factor), 0, 255)
            return np.take(lut, self.img).astype(np.uint8)

    def brightness(self, value: int) -> np.ndarray:
        if len(self.img.shape) == 2:  # grayscale image
            lut: np.ndarray = np.arange(256) + value
            return np.clip(lut[self.img], 0, 255).astype(np.uint8)

        lut: np.ndarray = np.arange(256) + value
        hsv = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.take(lut, hsv[:, :, 2].astype(np.int8)).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def saturation(self, percent: float) -> np.ndarray:
        lut: np.ndarray = np.arange(256) * percent
        hsv = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.take(lut, hsv[:, :, 1].astype(np.int8)).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def calculations(self, second_img: np.ndarray, operation: str) -> np.ndarray:
        second_img = cv2.resize(second_img, (self.img.shape[1], self.img.shape[0]))

        if operation == "sum":
            return cv2.add(self.img, second_img)
        elif operation == "subtraction":
            return cv2.subtract(self.img, second_img)
        elif operation == "multiplication":
            return cv2.multiply(self.img, second_img)

    def threshold(self, threshold):
        return ((self.img > threshold) * 255).astype(np.uint8)

    # NOTE: resources: https://learnopencv.com/otsu-thresholding-with-opencv/
    def otsu(self):
        hist, bin_edges = np.histogram(self.img, bins=256)

        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        threshold = bin_mids[:-1][np.argmax(inter_class_variance)]

        return self.threshold(threshold), threshold

    def canny(self, 
              blur_strength: int = 5, 
              high_threshold_ratio: float = 0.15, 
              low_threshold_ratio: float = 0.05, 
              strong_pixel: float = 255,
              weak_pixel: float = 75):
        # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
        # convolution class init

        # 0. img to grayscale

        img = self.desaturate()

        # 1. noise reduction with gaussian blur

        img_blured = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

        # 2. calculate gradients with sobel filters
        sobelx = cv2.Sobel(img_blured, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_blured, cv2.CV_64F, 0, 1, ksize=3)

        G = np.sqrt(sobelx ** 2 + sobely ** 2)
        theta = np.arctan2(sobely, sobelx)

        # 3. non-maximum suppression
        W, H = G.shape
        M = np.zeros((W, H), dtype=np.int32)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        for i in range(1, W - 1):
            for j in range(1, H - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = G[i, j + 1]
                        r = G[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = G[i + 1, j - 1]
                        r = G[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = G[i + 1, j]
                        r = G[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = G[i - 1, j - 1]
                        r = G[i + 1, j + 1]

                    if (G[i, j] >= q) and (G[i, j] >= r):
                        M[i, j] = G[i, j]
                    else:
                        M[i, j] = 0

                except IndexError as e:
                    pass

        # 4. double thresholding

        high_thresh = np.max(M) * high_threshold_ratio
        low_thresh = high_thresh * low_threshold_ratio

        res = np.zeros((W, H), dtype=np.int32)

        strong_i, strong_j = np.where(M >= high_thresh)

        weak_i, weak_j = np.where((M <= high_thresh) & (M >= low_thresh))

        res[strong_i, strong_j] = strong_pixel
        res[weak_i, weak_j] = weak_pixel

        # 5. hysteresis

        for i in range(1, W - 1):
            for j in range(1, H - 1):
                if res[i, j] == weak_pixel:
                    try:
                        if ((res[i + 1, j - 1] == strong_pixel) or (res[i + 1, j] == strong_pixel) or (
                                res[i + 1, j + 1] == strong_pixel)
                                or (res[i, j - 1] == strong_pixel) or (res[i, j + 1] == strong_pixel)
                                or (res[i - 1, j - 1] == strong_pixel) or (res[i - 1, j] == strong_pixel) or (
                                        res[i - 1, j + 1] == strong_pixel)):
                            res[i, j] = strong_pixel
                        else:
                            res[i, j] = 0
                    except IndexError as e:
                        pass

        return res.astype(np.uint8)


# NOTE: resources: https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
class Histogram:
    def __init__(self) -> None:
        self.histSize = 256
        self.histRange = (0, 256)
        self.hist_w = 280
        self.hist_h = 350
        self.bin_w = int(round(self.hist_w / self.histSize))

    def rgb(self, img: np.ndarray) -> np.ndarray:
        histImage = np.full((self.hist_h, self.hist_w, 3), 23, dtype=np.uint8)
        if len(img.shape) != 3:
            return histImage
        bgr = cv2.split(img)
        b_hist = cv2.calcHist(bgr, [0], None, [self.histSize], self.histRange, accumulate=False)
        g_hist = cv2.calcHist(bgr, [1], None, [self.histSize], self.histRange, accumulate=False)
        r_hist = cv2.calcHist(bgr, [2], None, [self.histSize], self.histRange, accumulate=False)

        cv2.normalize(b_hist, b_hist, alpha=0, beta=self.hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=self.hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=self.hist_h, norm_type=cv2.NORM_MINMAX)

        for i in range(1, self.histSize):
            cv2.line(histImage, (self.bin_w * (i - 1), self.hist_h - int(b_hist[i - 1])),
                     (self.bin_w * (i), self.hist_h - int(b_hist[i])),
                     (255, 0, 0), thickness=2)
            cv2.line(histImage, (self.bin_w * (i - 1), self.hist_h - int(g_hist[i - 1])),
                     (self.bin_w * (i), self.hist_h - int(g_hist[i])),
                     (0, 255, 0), thickness=2)
            cv2.line(histImage, (self.bin_w * (i - 1), self.hist_h - int(r_hist[i - 1])),
                     (self.bin_w * (i), self.hist_h - int(r_hist[i])),
                     (0, 0, 255), thickness=2)

        return histImage

    def grayscale(self, img: np.ndarray) -> np.ndarray:
        histImage = np.full((self.hist_h, self.hist_w, 3), 23, dtype=np.uint8)
        if len(img.shape) != 3:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [self.histSize], self.histRange, accumulate=False)
        cv2.normalize(gray_hist, gray_hist, alpha=0, beta=self.hist_h, norm_type=cv2.NORM_MINMAX)

        for i in range(1, self.histSize):
            cv2.line(histImage, (self.bin_w * (i - 1), self.hist_h - int(gray_hist[i - 1])),
                     (self.bin_w * (i), self.hist_h - int(gray_hist[i])),
                     (255, 255, 255), thickness=2)

        return histImage

    def stretch(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) != 3:
            lut = np.clip((256 / (np.max(img) - np.min(img))) * (np.arange(256) - np.min(img)), 0, 255)

            return np.take(lut, img).astype(np.uint8)
        else:
            bgr = cv2.split(img)
            stretched = []
            for channel in bgr:
                lut = np.clip(256 / (np.max(channel) - np.min(channel)) * (np.arange(256) - np.min(channel)), 0, 255)
                stretched.append(np.take(lut, channel).astype(np.uint8))

            return np.array(stretched).transpose([1, 2, 0])

    # NOTE resources: https://towardsdatascience.com/histogram-equalization-a-simple-way-to-improve-the-contrast-of-your-image-bcd66596d815
    def equalize(self, img: np.ndarray):
        if (img.shape) == 3:
            b, g, r = cv2.split(img)
            channels = [b, g, r]
            for i, channel in enumerate(channels):
                hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
                cdf = hist.cumsum() / hist.sum()
                cdf_normalized = (cdf - cdf.min()) / (1 - cdf.min())
                cdf_mapped = (cdf_normalized * 255).astype('uint8')
                channels[i] = cdf_mapped[channel]

            return cv2.merge(channels)
        else:
            h_gr, _ = np.histogram(img.flatten(), 256, [0, 256])
            cdf_gr = np.cumsum(h_gr)
            cdf_m_gr = np.ma.masked_equal(cdf_gr, 0)
            cdf_m_gr = (cdf_m_gr - cdf_m_gr.min()) * 255 / (cdf_m_gr.max() - cdf_m_gr.min())
            cdf_final_b = np.ma.filled(cdf_m_gr, 0).astype('uint8')

            return cdf_final_b[img]


class Conv:
    def __init__(self, kernel):
        self.kernel = np.array(kernel)

    def uniform_blur(self, image: np.ndarray, kernel_size: int = 3):
        return cv2.blur(image, (kernel_size, kernel_size))

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 3, sigma: int = 0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def sharpen(self, image: np.ndarray, kernel_size: int = 3, sigma: int = 0):
        blurred = self.gaussian_blur(image, kernel_size, sigma)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def edge_detection(self,
                       image: np.ndarray,
                       id: str,
                       kernel_size: int = 3,
                       sigma: int = 0):
        if id == 'sobel':
            return self._sobel(image)
        elif id == 'previtt':
            return self._previtt(image)
        elif id == 'roberts':
            return self._roberts(image)
        elif id == 'laplacian':
            return self._laplacian(image)
        elif id == 'log':
            return self._log(image, kernel_size, sigma)

    def custom(self, image: np.ndarray, custom_kernel):
        return cv2.filter2D(src=image, ddepth=-1, kernel=custom_kernel)

    def _sobel(self, image: np.ndarray):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(sobelx)
        abs_grad_y = cv2.convertScaleAbs(sobely)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        return grad

    def _previtt(self, image: np.ndarray):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        filter_x = cv2.filter2D(image, -1, kernel_x)
        filter_y = cv2.filter2D(image, -1, kernel_y)

        return cv2.addWeighted(filter_x, 0.5, filter_y, 0.5, 0).astype(np.uint8)

    def _roberts(self, image: np.ndarray):
        kernel_x = np.array([[0, 1], [-1, 0]])
        kernel_y = np.array([[1, 0], [0, -1]])

        filter_x = cv2.filter2D(image, -1, kernel_x)
        filter_y = cv2.filter2D(image, -1, kernel_y)

        return cv2.addWeighted(filter_x, 0.5, filter_y, 0.5, 0).astype(np.uint8)

    def _laplacian(self, image: np.ndarray):
        return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)

    def _log(self, image: np.ndarray, kernel_size: int = 3, sigma: int = 0):
        blur = self.gaussian_blur(image, kernel_size, sigma)
        return cv2.Laplacian(blur, cv2.CV_64F).astype(np.uint8)
















