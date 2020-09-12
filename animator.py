import cv2 as cv
import numpy as np

import utils


class Animator:
    cur_tilt = 0
    cur_l_brow = 0
    cur_r_brow = 0
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

    def animate(self, angle, l_brow_pos, r_brow_pos):
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
        if l_brow_pos < self.cur_l_brow:
            self.cur_l_brow -= (self.cur_l_brow - l_brow_pos) / 2
        elif l_brow_pos > self.cur_l_brow:
            self.cur_l_brow += (l_brow_pos - self.cur_l_brow) / 2
        if r_brow_pos < self.cur_r_brow:
            self.cur_r_brow -= (self.cur_r_brow - r_brow_pos) / 2
        elif r_brow_pos > self.cur_r_brow:
            self.cur_r_brow += (r_brow_pos - self.cur_r_brow) / 2
        # лимиты сдвига
        if self.cur_l_brow < -20:
            self.cur_l_brow = 20
        elif self.cur_l_brow > 20:
            self.cur_l_brow = 20
        if self.cur_r_brow < -20:
            self.cur_r_brow = 20
        elif self.cur_r_brow > 20:
            self.cur_r_brow = 20

    def make_eye(self, eye, eye_white, pupil_pos, new_shape, pupil):
        pupil_shift = np.float32([[1, 0, pupil_pos], [0, 1, 0]])
        eye_pupil = cv.warpAffine(pupil, pupil_shift, new_shape, borderMode=cv.BORDER_REPLICATE)
        eye_pupil = cv.copyTo(eye_pupil, eye_white)
        res_eye = utils.blend_transparent(eye_white, eye_pupil)
        res_eye = utils.blend_transparent(res_eye, eye)
        return res_eye

    def put_mask(self, mouth_shape, r_eye_s, l_eye_s, r_pupil_pos, l_pupil_pos):
        repl = cv.BORDER_REPLICATE
        linear = cv.INTER_LINEAR
        rot = cv.getRotationMatrix2D((self.head.shape[0] / 2., self.head.shape[1] / 2. + 70), self.cur_tilt, 1)
        new_shape = (self.head.shape[1], self.head.shape[0])

        rot_hair_back = cv.warpAffine(self.hair_back, rot, new_shape, flags=linear, borderMode=repl)

        l_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_l_brow]])
        r_brow_shift = np.float32([[1, 0, 0], [0, 1, -self.cur_r_brow]])
        l_brow = cv.warpAffine(self.l_brow, l_brow_shift, new_shape, borderMode=repl)
        r_brow = cv.warpAffine(self.r_brow, r_brow_shift, new_shape, borderMode=repl)

        self.res = rot_hair_back
        self.res = utils.blend_transparent(self.res, self.background)

        face = utils.blend_transparent(self.head, self.mouth[mouth_shape])  # голова + рот
        brows = cv.bitwise_or(r_brow, l_brow)
        face = utils.blend_transparent(face, brows)  # голова + рот + брови

        r_eye = self.make_eye(self.r_eye[r_eye_s], self.r_eye_white[r_eye_s], r_pupil_pos, new_shape, self.r_eye_pupil)
        l_eye = self.make_eye(self.l_eye[l_eye_s], self.l_eye_white[l_eye_s], l_pupil_pos, new_shape, self.l_eye_pupil)
        eyes = cv.bitwise_or(r_eye, l_eye)

        face = utils.blend_transparent(face, eyes)  # голова + рот + брови + глаза
        face = utils.blend_transparent(face, self.hair)  # голова + рот + брови + глаза + волосы
        face = cv.warpAffine(face, rot, new_shape, flags=linear, borderMode=repl)
        face = cv.cvtColor(face, cv.COLOR_BGR2BGRA)

        self.res = utils.blend_transparent(self.res, face)

    def display(self):
        cv.imshow("Animezator", self.res)
