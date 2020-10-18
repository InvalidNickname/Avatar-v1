import math
import cv2 as cv
import cupy as cp

from limits import *


def length(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def blend_transparent(background, overlay):
    overlay_img = overlay[:, :, :3]  # 3 ch
    overlay_mask = overlay[:, :, 3]  # 1 ch
    background_img = background[:, :, :3]  # 3 ch
    background_mask = 255 - overlay_mask  # 1 ch

    b_m = background[:, :, 3]
    mask = cv.add(overlay_mask.get(), b_m.get())  # 1 ch

    overlay_mask = cp.array(cv.cvtColor(overlay_mask.get(), cv.COLOR_GRAY2BGR))  # 3 ch
    background_mask = cp.array(cv.cvtColor(background_mask.get(), cv.COLOR_GRAY2BGR))  # 3 ch

    background_part = background_img * (background_mask / 255.0)  # 3 ch
    overlay_part = overlay_img * (overlay_mask / 255.0)  # 3 ch

    ch_3_res = cp.add(background_part, overlay_part)
    res = cp.dstack([ch_3_res, mask])
    return cp.array(res, dtype=cp.uint8)


def vertical_shift(arr, num, fill_value=0):
    result = cp.empty_like(arr)
    if num > 0:
        result[:num, :, :] = fill_value
        result[num:, :, :] = arr[:-num, :, :]
    elif num < 0:
        result[num:, :, :] = fill_value
        result[:num, :, :] = arr[-num:, :, :]
    else:
        result = arr
    return result


def horizontal_shift(arr, num, fill_value=0):
    result = cp.empty_like(arr)
    if num > 0:
        result[:, :num, :] = fill_value
        result[:, num:, :] = arr[:, :-num, :]
    elif num < 0:
        result[:, num:, :] = fill_value
        result[:, :num, :] = arr[:, -num:, :]
    else:
        result = arr
    return result


def get_rot_mat(center, angle):
    angle = math.radians(angle)
    alpha = math.cos(angle)
    beta = math.sin(angle)
    res = cp.array([[alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                    [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])
    return res


def load_image(path):
    img = cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED), (0, 0), fy=1 / DOWNSCALING, fx=1 / DOWNSCALING)
    return cp.asarray(img)
