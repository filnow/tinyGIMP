from typing import List
import cv2
import numpy as np


class ImageLoader:
    #NOTE: We use static methods because we don't need to instantiate the class
    @staticmethod
    def load(path: str) -> np.ndarray:
        image_type = ""
        lines = [line.replace("\n", "") for line in open(path, encoding='latin1').readlines()]
        for idx, line in enumerate(lines):
            if line[0] == "#":
                continue
            elif line[0] == "P":
                image_type = line
            else:
                break
        size = [int(x) for x in lines[idx].split()] if image_type != "" else []
        image = ImageLoader._get_image(lines, size, idx, path, image_type)
        return image

    @staticmethod
    def _get_image(lines: List[str],
                   size: List[int],  
                   idx: int, 
                   path: str, 
                   image_type: str) -> np.ndarray:
        if image_type == 'P1':
            print(lines[idx+1:])
            image = np.asarray([(int(x)-1)*-255 for row in lines[idx+1:] for x in row.replace(' ', '')]).reshape(size)
        elif image_type == 'P2':
            image = np.asarray([int(x) for x in lines[idx+2:]]).reshape(size)
        elif image_type == 'P3':
            image = np.asarray([int(x) for x in lines[idx+2:]]).reshape(*size, 3)
            #NOTE: Swap the red and blue channels
            image[:, :, [0, 2]] = image[:, :, [2, 0]]
        else:
            image = cv2.imread(path)
        return image
    

class ImageProcessor:
    def __init__(self, img: np.ndarray) -> None:
        self.img = img
    
    def desaturate(self) -> np.ndarray:
        return cv2.cvtColor(self.img.astype(np.float32), cv2.COLOR_BGR2GRAY)

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
        lut: np.ndarray = np.arange(256) + value
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.take(lut, hsv[:, :, 2].astype(np.int8)).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def saturation(self, percent: float) -> np.ndarray:
        lut: np.ndarray = np.arange(256) * percent
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV).astype(np.float32)
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
        
    #NOTE: monochrome should be a interface to drawing a functions        
    def monochrome(self) -> np.ndarray:
        #NOTE: We use the YIQ transformation
        #https://en.wikipedia.org/wiki/YIQ
        yiq = np.dot(self.img, [[0.299, 0.587, 0.114], 
                                [-0.596, -0.275, 0.321], 
                                [0.212, -0.523, 0.311]])
        return yiq.astype(np.uint8)

