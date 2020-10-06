import cv2 as cv
import random
import math

from overlay import *
from limits import *


class Animator:
    cur_tilt = 0
    cur_l_brow = 0
    cur_r_brow = 0
    cur_r_pupil = 0
    cur_l_pupil = 0
    cur_r_eye_shape = 3
    cur_l_eye_shape = 3
    blinking = 0  # 0 - нет, -1 - закрывает глаза, 1 - открывает глаза
    head_tilt = 0
    target_head_tilt = 0
    cur_head_offset = 0
    head_central_y = -1
    res = None

    def __init__(self, overlays):
        self.imgs = Overlay(overlays)

    def blink(self):
        if self.blinking == 1:
            if self.cur_r_eye_shape < 3:
                self.cur_r_eye_shape += 1
            else:
                self.blinking = 0
            if self.cur_l_eye_shape < 3:
                self.cur_l_eye_shape += 1
            else:
                self.blinking = 0
        else:
            if self.blinking == 0:
                self.blinking = -1
            if self.cur_r_eye_shape > 0:
                self.cur_r_eye_shape -= 1
            else:
                self.blinking = 1
                self.cur_r_eye_shape += 1
            if self.cur_l_eye_shape > 0:
                self.cur_l_eye_shape -= 1
            else:
                self.blinking = 1
                self.cur_l_eye_shape += 1

    def animate(self, angle, l_brow_pos, r_brow_pos, r_pupil_pos, l_pupil_pos, target_head_offset):
        self.head_tilt = angle
        # сдвигаем лицо по вертикали
        self.cur_head_offset = move_slowly(target_head_offset, self.cur_head_offset, 4)
        # поворачиваем лицо
        self.cur_tilt = move_slowly(self.head_tilt, self.cur_tilt, 3)
        self.cur_tilt = set_limits(self.cur_tilt, LIMIT_HEAD_TILT, -LIMIT_HEAD_TILT)
        # двигаем брови
        self.cur_l_brow = move_slowly(l_brow_pos, self.cur_l_brow, 2)
        self.cur_l_brow = set_limits(self.cur_l_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        self.cur_r_brow = move_slowly(r_brow_pos, self.cur_r_brow, 2)
        self.cur_r_brow = set_limits(self.cur_r_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        # двигаем зрачки
        self.cur_r_pupil = move_slowly(r_pupil_pos, self.cur_r_pupil, 2)
        self.cur_l_pupil = move_slowly(l_pupil_pos, self.cur_l_pupil, 2)

    def standby_animate(self):
        if abs(self.target_head_tilt - self.head_tilt) < 0.05:
            self.target_head_tilt = random.random() * 20 - 10
        self.head_tilt = move_slowly(self.target_head_tilt, self.head_tilt, 7)
        self.animate(self.head_tilt, 0, 0, 0, 0, self.head_central_y)

    def standby(self, animate):
        if animate:
            self.standby_animate()
        if self.blinking == 0:
            self.cur_r_eye_shape = 3
            self.cur_l_eye_shape = 3
        else:
            self.blink()
        self.put_mask(0, 3, 3)
        self.display()

    def put_mask(self, mouth_shape, r_eye_s, l_eye_s):
        bmode = cv.BORDER_REPLICATE

        head_offset = (self.cur_head_offset - self.head_central_y) * HEAD_MAX_Y_OFFSET / 250

        head_shift = np.float32([[1, 0, 0], [0, 1, head_offset]])
        rot = cv.getRotationMatrix2D(
            (self.imgs.shape()[0] / 2, self.imgs.shape()[1] / 2 + HEAD_ROT_POINT_Y), self.cur_tilt, 1)
        rot = np.vstack([rot, [0, 0, 1]])
        rot = np.matmul(head_shift, rot)

        rot_hair_back = cv.warpAffine(self.imgs.get_img("hair_back"), rot, self.imgs.warp_shape(),
                                      flags=cv.INTER_LINEAR, borderMode=bmode)

        l_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_l_brow]])
        l_brow = cv.warpAffine(self.imgs.get_img("l_brow"), l_brow_shift, self.imgs.warp_shape(), borderMode=bmode)

        r_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_r_brow]])
        r_brow = cv.warpAffine(self.imgs.get_img("r_brow"), r_brow_shift, self.imgs.warp_shape(), borderMode=bmode)

        self.res = rot_hair_back

        self.res = utils.blend_transparent(self.res, self.imgs.get_img("background"))

        face = self.imgs.get_img("head")
        face = utils.blend_transparent(face, self.imgs.get_img("mouth_" + str(mouth_shape)))  # голова + рот
        brows = cv.bitwise_or(r_brow, l_brow)
        face = utils.blend_transparent(face, brows)  # голова + рот + брови

        if self.blinking == 0:
            self.cur_r_eye_shape = r_eye_s
            self.cur_l_eye_shape = l_eye_s
        else:
            self.blink()
        r_eye = make_eye(self.cur_r_pupil, self.imgs, self.cur_r_eye_shape, "r")
        l_eye = make_eye(self.cur_l_pupil, self.imgs, self.cur_l_eye_shape, "l")
        eyes = cv.bitwise_or(r_eye, l_eye)

        face = utils.blend_transparent(face, eyes)  # голова + рот + брови + глаза

        face = utils.blend_transparent(face, self.imgs.get_img("hair"))  # голова + рот + брови + глаза + волосы
        face = cv.warpAffine(face, rot, self.imgs.warp_shape(), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        face = cv.cvtColor(face, cv.COLOR_BGR2BGRA)

        self.res = utils.blend_transparent(self.res, face)

    def display(self):
        cv.imshow("Animezator", self.res)

    def change_overlay(self, overlay_id):
        self.imgs.overlay_id = overlay_id


def make_eye(pupil_pos, imgs, eye_shape, l_r):
    pupil_shift = np.float32([[1, 0, pupil_pos], [0, 1, 0]])
    eye_pupil = cv.warpAffine(imgs.get_img(l_r + "_eye_pupil"), pupil_shift, imgs.warp_shape(),
                              borderMode=cv.BORDER_REPLICATE)
    eye_pupil = cv.copyTo(eye_pupil, imgs.get_img(l_r + "_eye_white_" + str(eye_shape)))
    res_eye = utils.blend_transparent(imgs.get_img(l_r + "_eye_white_" + str(eye_shape)), eye_pupil)
    res_eye = utils.blend_transparent(res_eye, imgs.get_img(l_r + "_eye_" + str(eye_shape)))
    return res_eye


def move_slowly(direction, current, multiplier):
    if direction < current:
        current -= (current - direction) / multiplier
    elif direction > current:
        current += (direction - current) / multiplier
    return current


def set_limits(current, upper_limit, lower_limit):
    if current > upper_limit:
        current = upper_limit
    elif current < lower_limit:
        current = lower_limit
    return current
