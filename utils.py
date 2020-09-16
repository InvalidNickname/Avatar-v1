import math
import cv2 as cv
import numpy as np


def length(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def blend_transparent(background, overlay):
    overlay_img = overlay[:, :, :3]  # 3 ch
    overlay_mask = overlay[:, :, 3]  # 1 ch

    background_img = background[:, :, :3]  # 3 ch
    background_mask = 255 - overlay_mask  # 1 ch

    mask = cv.add(overlay_mask, background[:, :, 3])  # 1 ch

    overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)  # 3 ch
    background_mask = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)  # 3 ch

    background_part = background_img * (background_mask / 255.0)  # 3 ch
    overlay_part = overlay_img * (overlay_mask / 255.0)  # 3 ch

    b, g, r = cv.split(np.uint8(cv.add(background_part, overlay_part)))

    return cv.merge((b, g, r, mask))


def load_image(path):
    return cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED), (0, 0), fy=0.5, fx=0.5)
