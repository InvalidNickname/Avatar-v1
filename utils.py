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


def get_rot_mat(center, angle):
    angle = math.radians(angle)
    alpha = math.cos(angle)
    beta = math.sin(angle)
    res = cp.array([[alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                    [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]])
    return res


def get_shift_mat(horizontal=0, vertical=0):
    return cp.array([[1, 0, horizontal], [0, 1, vertical]], dtype=cp.float32)


def warp_affine(src, mat):
    h, w = src.shape[:2]
    dst_y, dst_x = cp.indices((h, w))
    dst_lin_hmg_pts = cp.stack((dst_x.ravel(), dst_y.ravel(), cp.ones(dst_y.size)))
    new_pos = cp.dot(mat, dst_lin_hmg_pts)
    new_pos = cp.where(new_pos < 0, 0, new_pos)
    new_pos[0] = cp.where(new_pos[0] >= w, 0, new_pos[0])
    new_pos[1] = cp.where(new_pos[1] >= h, 0, new_pos[1])
    new_pos = cp.where(cp.abs(new_pos % 1) >= 0.5, cp.ceil(new_pos), cp.floor(new_pos)).astype(int)
    dst = cp.zeros(src.shape, dtype=cp.uint8)
    dst[new_pos[1], new_pos[0]] = src.reshape(-1, 4)
    return dst


def load_image(path):
    if DOWNSCALING != 1:
        img = cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED), (0, 0), fy=1 / DOWNSCALING, fx=1 / DOWNSCALING)
    else:
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
    return cp.asarray(img)
