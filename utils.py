import math
import cv2 as cv
import cupy as cp
import numpy as np
from datetime import datetime

from limits import *


def length(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def blend_transparent(background, overlay):
    overlay_img = overlay[..., :3]  # 3 ch
    overlay_mask = overlay[..., 3]  # 1 ch
    background_img = background[..., :3]  # 3 ch
    background_mask = cp.invert(overlay_mask)  # 1 ch

    b_m = background[..., 3]  # 1 ch
    mask = cp.add(overlay_mask.astype(cp.uint16), b_m.astype(cp.uint16))  # 1 ch
    mask = cp.where(mask > 255, 255, mask)  # 1 ch

    overlay_mask = cp.divide(overlay_mask, 255.0)  # 1 ch
    overlay_mask = cp.dstack([overlay_mask, overlay_mask, overlay_mask])  # 3 ch
    background_mask = cp.divide(background_mask, 255.0)  # 1 ch
    background_mask = cp.dstack([background_mask, background_mask, background_mask])  # 3 ch

    background_part = background_img * background_mask  # 3 ch
    overlay_part = overlay_img * overlay_mask  # 3 ch

    ch_3_res = cp.add(background_part, overlay_part)
    res = cp.dstack([ch_3_res, mask])
    return res.astype(cp.uint8)


def speed_test(tag, method):
    sumt = 0
    for i in range(1000):
        st = datetime.now().microsecond
        method()
        sumt += datetime.now().microsecond - st
    print(tag, "med:", sumt / 1000)


def warp_with_bb(img, bb, matrix, shape):
    img_part = img[bb[1]:bb[3], bb[0]:bb[2], :]
    img_part = cv.warpAffine(img_part, matrix, shape, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    img[bb[1]:bb[3], bb[0]:bb[2], :] = img_part


def vertical_shift(arr, num, fill_value=0):
    result = cp.empty_like(arr)
    if num > 1:
        result[:num, :, :] = fill_value
        result[num:, :, :] = arr[:-num, :, :]
    elif num < -1:
        result[num:, :, :] = fill_value
        result[:num, :, :] = arr[-num:, :, :]
    else:
        result = arr
    return result


def shift(arr, vertical=0, horizontal=0, fill_value=0):
    result = cp.empty_like(arr)
    if horizontal >= 1:
        if vertical >= 1:
            result[:, :horizontal, :] = fill_value
            result[:vertical, :, :] = fill_value
            result[vertical:, horizontal:, :] = arr[:-vertical, :-horizontal, :]
        elif vertical <= -1:
            result[:, :horizontal, :] = fill_value
            result[vertical:, :, :] = fill_value
            result[:vertical, horizontal:, :] = arr[-vertical:, :-horizontal, :]
        else:
            result = horizontal_shift(arr, horizontal, fill_value)
    elif horizontal <= -1:
        if vertical >= 1:
            result[:, horizontal:, :] = fill_value
            result[:vertical, :, :] = fill_value
            result[vertical:, :horizontal, :] = arr[:-vertical, -horizontal:, :]
        elif vertical <= -1:
            result[:, horizontal:, :] = fill_value
            result[vertical:, :, :] = fill_value
            result[:vertical, :horizontal, :] = arr[-vertical:, -horizontal:, :]
        else:
            result = horizontal_shift(arr, horizontal, fill_value)
    else:
        if vertical >= 1 or vertical <= -1:
            result = vertical_shift(arr, vertical, fill_value)
        else:
            result = arr.copy()
    return result


def horizontal_shift(arr, num, fill_value=0):
    result = cp.empty_like(arr)
    if num > 1:
        result[:, :num, :] = fill_value
        result[:, num:, :] = arr[:, :-num, :]
    elif num < -1:
        result[:, num:, :] = fill_value
        result[:, :num, :] = arr[:, -num:, :]
    else:
        result = arr
    return result


def get_rot_mat(center, angle, cpu=False):
    angle = math.radians(angle)
    alpha = math.cos(angle)
    beta = math.sin(angle)
    if cpu:
        res = np.array([[alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])
    else:
        res = cp.array([[alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])
    return res


def get_shift_mat(horizontal=0, vertical=0):
    return cp.float32([[1, 0, horizontal], [0, 1, vertical]])


def load_image(path):
    img = cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED), (0, 0), fy=1 / DOWNSCALING, fx=1 / DOWNSCALING)
    return cp.asarray(img)
