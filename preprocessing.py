import numpy as np
import cv2

def Old_Preprocessor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    resized = cv2.resize(edges, (256, 256))
    return resized

def Preprocessor(image_path):
    t_lower = 30 # lower threshold
    t_upper = 100 # upper threshold
    aperture_size = 3
    L2Gradient = True
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    resized = cv2.resize(blur, (256, 256))
    edges = cv2.Canny(resized, t_lower, t_upper, 
                      apertureSize= aperture_size,
                      L2gradient = L2Gradient)
    img_normalized = cv2.normalize(edges, None, 0, 1.0, cv2.NORM_MINMAX)
    return img_normalized
