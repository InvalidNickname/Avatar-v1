import cv2 as cv
import numpy as np


class Animator:
    cur_tilt = 0
    res = None

    def __init__(self):
        background = cv.imread("data/animation/background.png", cv.IMREAD_GRAYSCALE)
        self.background = cv.resize(background, (0, 0), fx=0.5, fy=0.5)

        head = cv.imread("data/animation/head.png", cv.IMREAD_GRAYSCALE)
        self.head = cv.resize(head, (0, 0), fx=0.5, fy=0.5)

        hair_back = cv.imread("data/animation/hair_back.png", cv.IMREAD_GRAYSCALE)
        self.hair_back = cv.resize(hair_back, (0, 0), fx=0.5, fy=0.5)

    def animate(self, angle):
        # поворачиваем лицо
        if angle < self.cur_tilt:
            self.cur_tilt -= (self.cur_tilt - angle) / 3
        elif angle > self.cur_tilt:
            self.cur_tilt += (angle - self.cur_tilt) / 3

        if self.cur_tilt > 15:
            self.cur_tilt = 15
        elif self.cur_tilt < -15:
            self.cur_tilt = -15
        #

    def put_mask(self):
        rot = cv.getRotationMatrix2D((self.head.shape[0] / 2., self.head.shape[1] / 2. + 70), self.cur_tilt, 1)

        rot_head = cv.warpAffine(self.head, rot, (self.head.shape[1], self.head.shape[0]), flags=cv.INTER_LINEAR,
                                 borderMode=cv.BORDER_REPLICATE)
        rot_hair_back = cv.warpAffine(self.hair_back, rot, (self.head.shape[1], self.head.shape[0]),
                                      flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

        self.res = rot_hair_back
        np.putmask(self.res, self.background > 0, self.background)
        np.putmask(self.res, rot_head > 0, rot_head)

    def display(self):
        cv.imshow("Animezator", self.res)
