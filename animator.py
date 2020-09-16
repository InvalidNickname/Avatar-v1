import cv2 as cv
import numpy as np

import utils
from limits import *


class Animator:
    cur_tilt = 0
    cur_l_brow = 0
    cur_r_brow = 0
    cur_r_pupil = 0
    cur_l_pupil = 0
    res = None

    def __init__(self):
        self.hair_back = utils.load_image("data/animation/hair_back.png")
        self.background = utils.load_image("data/animation/background.png")
        self.head = utils.load_image("data/animation/head.png")
        self.l_brow = utils.load_image("data/animation/left_brow.png")
        self.r_brow = utils.load_image("data/animation/right_brow.png")
        self.mouth = [utils.load_image("data/animation/mouth/1.png"),
                      utils.load_image("data/animation/mouth/2.png"),
                      utils.load_image("data/animation/mouth/3.png"),
                      utils.load_image("data/animation/mouth/4.png"),
                      # utils.load_image("data/animation/mouth/5.png")
                      ]
        self.r_eye_white = [utils.load_image("data/animation/right_eye/white_1.png"),
                            utils.load_image("data/animation/right_eye/white_2.png"),
                            utils.load_image("data/animation/right_eye/white_3.png"),
                            utils.load_image("data/animation/right_eye/white_4.png"),
                            # utils.load_image("data/animation/right_eye/white_2.png")
                            ]
        self.r_eye_pupil = utils.load_image("data/animation/right_eye/pupil.png")
        self.r_eye = [utils.load_image("data/animation/right_eye/eye_1.png"),
                      utils.load_image("data/animation/right_eye/eye_2.png"),
                      utils.load_image("data/animation/right_eye/eye_3.png"),
                      utils.load_image("data/animation/right_eye/eye_4.png"),
                      # utils.load_image("data/animation/right_eye/eye_2.png")
                      ]
        self.l_eye_white = [utils.load_image("data/animation/left_eye/white_1.png"),
                            utils.load_image("data/animation/left_eye/white_2.png"),
                            utils.load_image("data/animation/left_eye/white_3.png"),
                            utils.load_image("data/animation/left_eye/white_4.png"),
                            # utils.load_image("data/animation/left_eye/white_2.png")
                            ]
        self.l_eye_pupil = utils.load_image("data/animation/left_eye/pupil.png")
        self.l_eye = [utils.load_image("data/animation/left_eye/eye_1.png"),
                      utils.load_image("data/animation/left_eye/eye_2.png"),
                      utils.load_image("data/animation/left_eye/eye_3.png"),
                      utils.load_image("data/animation/left_eye/eye_4.png"),
                      # utils.load_image("data/animation/left_eye/eye_2.png")
                      ]
        self.hair = utils.load_image("data/animation/hair.png")

    def animate(self, angle, l_brow_pos, r_brow_pos, r_pupil_pos, l_pupil_pos):
        # поворачиваем лицо
        self.cur_tilt = move_slowly(angle, self.cur_tilt, 3)
        self.cur_tilt = set_limits(self.cur_tilt, LIMIT_HEAD_TILT, -LIMIT_HEAD_TILT)
        # двигаем брови
        self.cur_l_brow = move_slowly(l_brow_pos, self.cur_l_brow, 2)
        self.cur_l_brow = set_limits(self.cur_l_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        self.cur_r_brow = move_slowly(r_brow_pos, self.cur_r_brow, 2)
        self.cur_r_brow = set_limits(self.cur_r_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        # двигаем зрачки
        self.cur_r_pupil = move_slowly(r_pupil_pos, self.cur_r_pupil, 2)
        self.cur_l_pupil = move_slowly(l_pupil_pos, self.cur_l_pupil, 2)

    def put_mask(self, mouth_shape, r_eye_s, l_eye_s):
        rot = cv.getRotationMatrix2D((self.head.shape[0] / 2, self.head.shape[1] / 2 + HEAD_ROT_POINT_Y), self.cur_tilt,
                                     1)
        new_shape = (self.head.shape[1], self.head.shape[0])

        rot_hair_back = cv.warpAffine(self.hair_back, rot, new_shape, flags=cv.INTER_LINEAR,
                                      borderMode=cv.BORDER_REPLICATE)

        l_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_l_brow]])
        r_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_r_brow]])
        l_brow = cv.warpAffine(self.l_brow, l_brow_shift, new_shape, borderMode=cv.BORDER_REPLICATE)
        r_brow = cv.warpAffine(self.r_brow, r_brow_shift, new_shape, borderMode=cv.BORDER_REPLICATE)

        self.res = rot_hair_back
        self.res = utils.blend_transparent(self.res, self.background)

        face = utils.blend_transparent(self.head, self.mouth[mouth_shape])  # голова + рот
        brows = cv.bitwise_or(r_brow, l_brow)
        face = utils.blend_transparent(face, brows)  # голова + рот + брови

        r_eye = make_eye(self.r_eye[r_eye_s], self.r_eye_white[r_eye_s], self.cur_r_pupil, new_shape, self.r_eye_pupil)
        l_eye = make_eye(self.l_eye[l_eye_s], self.l_eye_white[l_eye_s], self.cur_l_pupil, new_shape, self.l_eye_pupil)
        eyes = cv.bitwise_or(r_eye, l_eye)

        face = utils.blend_transparent(face, eyes)  # голова + рот + брови + глаза
        face = utils.blend_transparent(face, self.hair)  # голова + рот + брови + глаза + волосы
        face = cv.warpAffine(face, rot, new_shape, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        face = cv.cvtColor(face, cv.COLOR_BGR2BGRA)

        self.res = utils.blend_transparent(self.res, face)

    def display(self):
        cv.imshow("Animezator", self.res)


def make_eye(eye, eye_white, pupil_pos, new_shape, pupil):
    pupil_shift = np.float32([[1, 0, pupil_pos], [0, 1, 0]])
    eye_pupil = cv.warpAffine(pupil, pupil_shift, new_shape, borderMode=cv.BORDER_REPLICATE)
    eye_pupil = cv.copyTo(eye_pupil, eye_white)
    res_eye = utils.blend_transparent(eye_white, eye_pupil)
    res_eye = utils.blend_transparent(res_eye, eye)
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
