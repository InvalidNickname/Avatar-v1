import cv2 as cv
import numpy as np

import utils


class Animator:
    cur_tilt = 0
    cur_left_brow = 0
    cur_right_brow = 0
    res = None

    def __init__(self):
        self.hair_back = utils.load_image("data/animation/hair_back.png")
        self.background = utils.load_image("data/animation/background.png")
        self.head = utils.load_image("data/animation/head.png")
        self.left_brow = utils.load_image("data/animation/left_brow.png")
        self.right_brow = utils.load_image("data/animation/right_brow.png")
        self.mouth = [utils.load_image("data/animation/mouth/1.png"),
                      utils.load_image("data/animation/mouth/2.png"),
                      utils.load_image("data/animation/mouth/3.png"),
                      utils.load_image("data/animation/mouth/4.png"),
                      # utils.load_image("data/animation/mouth/5.png")]
                      ]
        self.hair = utils.load_image("data/animation/hair.png")

    def animate(self, angle, left_brow_pos, right_brow_pos):
        # поворачиваем лицо
        if angle < self.cur_tilt:
            self.cur_tilt -= (self.cur_tilt - angle) / 3
        elif angle > self.cur_tilt:
            self.cur_tilt += (angle - self.cur_tilt) / 3
        # лимиты поворота
        if self.cur_tilt > 15:
            self.cur_tilt = 15
        elif self.cur_tilt < -15:
            self.cur_tilt = -15
        # двигаем брови
        if left_brow_pos < self.cur_left_brow:
            self.cur_left_brow -= (self.cur_left_brow - left_brow_pos) / 2
        elif left_brow_pos > self.cur_left_brow:
            self.cur_left_brow += (left_brow_pos - self.cur_left_brow) / 2
        if right_brow_pos < self.cur_right_brow:
            self.cur_right_brow -= (self.cur_right_brow - right_brow_pos) / 2
        elif right_brow_pos > self.cur_right_brow:
            self.cur_right_brow += (right_brow_pos - self.cur_right_brow) / 2
        # лимиты сдвига
        if self.cur_left_brow < -20:
            self.cur_left_brow = 20
        elif self.cur_left_brow > 20:
            self.cur_left_brow = 20
        if self.cur_right_brow < -20:
            self.cur_right_brow = 20
        elif self.cur_right_brow > 20:
            self.cur_right_brow = 20

    def put_mask(self, mouth_shape):
        rot = cv.getRotationMatrix2D((self.head.shape[0] / 2., self.head.shape[1] / 2. + 70), self.cur_tilt, 1)
        new_shape = (self.head.shape[1], self.head.shape[0])

        rot_head = cv.warpAffine(self.head, rot, new_shape, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        rot_hair_back = cv.warpAffine(self.hair_back, rot, new_shape, flags=cv.INTER_LINEAR,
                                      borderMode=cv.BORDER_REPLICATE)
        rot_mouth = cv.warpAffine(self.mouth[mouth_shape], rot, new_shape, flags=cv.INTER_LINEAR,
                                  borderMode=cv.BORDER_REPLICATE)
        rot_hair = cv.warpAffine(self.hair, rot, new_shape, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

        left_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_left_brow]])
        right_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_right_brow]])

        rot_left_brow = cv.warpAffine(self.left_brow, left_brow_shift, new_shape, borderMode=cv.BORDER_REPLICATE)
        rot_left_brow = cv.warpAffine(rot_left_brow, rot, new_shape, flags=cv.INTER_LINEAR,
                                      borderMode=cv.BORDER_REPLICATE)
        rot_right_brow = cv.warpAffine(self.right_brow, right_brow_shift, new_shape, borderMode=cv.BORDER_REPLICATE)
        rot_right_brow = cv.warpAffine(rot_right_brow, rot, new_shape, flags=cv.INTER_LINEAR,
                                       borderMode=cv.BORDER_REPLICATE)

        self.res = cv.cvtColor(rot_hair_back, cv.COLOR_BGRA2BGR)
        self.res = utils.blend_transparent(self.res, self.background)
        self.res = utils.blend_transparent(self.res, rot_head)
        self.res = utils.blend_transparent(self.res, rot_mouth)
        brows = cv.bitwise_or(rot_right_brow, rot_left_brow)
        self.res = utils.blend_transparent(self.res, brows)
        self.res = utils.blend_transparent(self.res, rot_hair)

    def display(self):
        cv.imshow("Animezator", self.res)
