import math
import cv2 as cv
import numpy as np

from limits import *


def length(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def blend_partial(background, overlay):
    overlay_img = overlay[:, :, :3]  # 3 ch
    overlay_mask = overlay[:, :, 3]  # 1 ch
    if np.max(overlay_mask) == 0:
        return background
    else:
        background_img = background[:, :, :3]  # 3 ch
        background_mask = 255 - overlay_mask  # 1 ch

        print(overlay_mask.shape, " ", background[:, :, 3].shape)

        mask = cv.add(overlay_mask, background[:, :, 3])  # 1 ch

        overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)  # 3 ch
        background_mask = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)  # 3 ch

        background_part = background_img * (background_mask / 255.0)  # 3 ch
        overlay_part = overlay_img * (overlay_mask / 255.0)  # 3 ch

        b, g, r = cv.split(np.uint8(cv.add(background_part, overlay_part)))
        return cv.merge((b, g, r, mask))


def blend_transparent(background, overlay):
    height = background.shape[1]
    n = 12
    res = np.zeros(background.shape, dtype=np.uint8)
    for i in range(n):
        shift = int(height * i / n)
        background_part = background[:, shift:int(height * (i + 1) / n), :]
        overlay_part = overlay[:, shift:int(height * (i + 1) / n), :]
        res[:, shift:int(height * (i + 1) / n), :] = blend_partial(background_part, overlay_part)
    return res


def load_image(path):
    return cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED), (0, 0), fy=1 / DOWNSCALING, fx=1 / DOWNSCALING)
