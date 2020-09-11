import math
import cv2 as cv
import numpy as np


def length(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def blend_transparent(background, overlay):
    overlay_img = overlay[:, :, :3]
    overlay_mask = overlay[:, :, 3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)
    background_mask = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)

    background_part = background / 255.0 * background_mask
    overlay_part = overlay_img / 255.0 * overlay_mask

    return np.uint8(cv.add(background_part, overlay_part))


def load_image(path):
    return cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED), (0, 0), fy=0.5, fx=0.5)
