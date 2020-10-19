import cv2 as cv
import random
import time

from overlay import *
from limits import *

bmode = cv.BORDER_REPLICATE
linear = cv.INTER_LINEAR


class Animator:
    cur_tilt = 0
    cur_l_brow = 0
    cur_r_brow = 0
    cur_r_brow_tilt = 0
    cur_l_brow_tilt = 0
    cur_r_pupil = 0
    cur_l_pupil = 0
    cur_r_eye_shape = 6
    cur_l_eye_shape = 6

    blinking = 0  # 0 - нет, -1 - закрывает глаза, 1 - открывает глаза
    blinking_mat = [0, 2, 3, 5]  # изображения глаз при моргании
    blinking_step = 4

    head_tilt = 0
    target_head_tilt = 0
    cur_head_offset = 0
    head_central_y = -1
    res = None

    cur_breathe = 0
    breathe_status = 1

    def __init__(self, overlays, animations):
        self.imgs = Overlay(overlays, animations)

    def blink(self):
        if self.blinking == 1:
            if self.blinking_step < 3:
                self.blinking_step += 1
            else:
                self.blinking = 0
        else:
            if self.blinking == 0:
                self.blinking = -1
                for i in range(len(self.blinking_mat)):
                    if self.cur_r_eye_shape < self.blinking_mat[len(self.blinking_mat) - i - 1]:
                        self.blinking_step = len(self.blinking_mat) - i - 1
            if self.blinking_step > 0:
                self.blinking_step -= 1
            else:
                self.blinking = 1
                self.blinking_step += 1
        self.cur_r_eye_shape = self.blinking_mat[self.blinking_step]
        self.cur_l_eye_shape = self.cur_r_eye_shape

    def animate(self, angle, l_brow_pos, r_brow_pos, r_pupil_pos, l_pupil_pos, target_head_offset, l_brow_tilt,
                r_brow_tilt):
        self.head_tilt = angle
        # сдвигаем лицо по вертикали
        self.cur_head_offset = move_slowly(target_head_offset, self.cur_head_offset, 4)
        # поворачиваем лицо
        self.cur_tilt = move_slowly(self.head_tilt, self.cur_tilt, 3)
        self.cur_tilt = set_limits(self.cur_tilt, LIMIT_HEAD_TILT, -LIMIT_HEAD_TILT)
        # двигаем брови
        self.cur_l_brow = move_slowly(l_brow_pos, self.cur_l_brow, 2)
        self.cur_l_brow = set_limits(self.cur_l_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        self.cur_l_brow_tilt = move_slowly(self.cur_l_brow_tilt, l_brow_tilt, 2)
        self.cur_r_brow = move_slowly(r_brow_pos, self.cur_r_brow, 2)
        self.cur_r_brow = set_limits(self.cur_r_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        self.cur_r_brow_tilt = move_slowly(self.cur_r_brow_tilt, r_brow_tilt, 2)
        # двигаем зрачки
        self.cur_r_pupil = move_slowly(r_pupil_pos, self.cur_r_pupil, 2)
        self.cur_l_pupil = move_slowly(l_pupil_pos, self.cur_l_pupil, 2)
        # дыхание
        self.cur_breathe, self.breathe_status = breathe(self.cur_breathe, self.breathe_status)

    def standby_animate(self):
        if abs(self.target_head_tilt - self.head_tilt) < 0.05:
            self.target_head_tilt = random.random() * 20 - 10
        self.head_tilt = move_slowly(self.target_head_tilt, self.head_tilt, 7)
        self.animate(self.head_tilt, 0, 0, 0, 0, self.head_central_y, 0, 0)

    def standby(self, animate):
        if animate:
            self.standby_animate()
        if self.blinking == 0:
            self.cur_r_eye_shape = 6
            self.cur_l_eye_shape = 6
        else:
            self.blink()
        self.put_mask(0, 6, 6)
        self.display()

    def put_mask(self, mouth_shape, r_eye_s, l_eye_s):

        head_offset = (self.cur_head_offset - self.head_central_y) * HEAD_MAX_Y_OFFSET / 250

        head_shift = cp.array([[1, 0, 0], [0, 1, head_offset]], dtype=cp.float32)
        rot_x = self.imgs.shape()[0] / 2
        rot_y = self.imgs.shape()[1] / 2 + HEAD_ROT_POINT_Y
        rot = utils.get_rot_mat((rot_x, rot_y), self.cur_tilt)
        rot = cp.vstack([rot, [0, 0, 1]])
        rot = cp.matmul(head_shift, rot).get()

        hair_back = self.imgs.get_img("hair_back").get()
        rot_hair_back = cv.warpAffine(hair_back, rot, self.imgs.w_s(), flags=linear, borderMode=bmode)

        l_brow_mat = make_brow_warp_matrix(self.cur_l_brow, L_BROW_ROT_X, L_BROW_ROT_Y, self.cur_l_brow_tilt)
        l_brow = self.imgs.get_img("l_brow").get()
        l_brow = cv.warpAffine(l_brow, l_brow_mat, self.imgs.w_s(), borderMode=bmode)

        r_brow_mat = make_brow_warp_matrix(self.cur_r_brow, R_BROW_ROT_X, R_BROW_ROT_Y, self.cur_r_brow_tilt)
        r_brow = self.imgs.get_img("r_brow").get()
        r_brow = cv.warpAffine(r_brow, r_brow_mat, self.imgs.w_s(), borderMode=bmode)

        body = cp.array(rot_hair_back)  # gpu
        body = utils.blend_transparent(body, self.imgs.get_img("body"))

        shadow = self.imgs.get_img("head_shadow").get()
        shadow = cv.warpAffine(shadow, rot, self.imgs.w_s(), flags=linear, borderMode=bmode)
        shadow = cp.bitwise_and(self.imgs.get_img("body"), cp.array(shadow))
        body = utils.blend_transparent(body, shadow)  # gpu

        face = self.imgs.get_img("head")  # gpu
        brows = cp.bitwise_or(cp.array(r_brow), cp.array(l_brow))  # брови gpu
        brow_mouth = cp.bitwise_or(brows, self.imgs.get_img("mouth_" + str(mouth_shape)))  # брови + рот gpu

        if self.blinking == 0:
            self.cur_r_eye_shape = r_eye_s
            self.cur_l_eye_shape = l_eye_s
        else:
            self.blink()

        eyes = make_eyes(self.cur_l_pupil, self.cur_r_pupil, self.imgs, self.cur_l_eye_shape, self.cur_r_eye_shape)

        brows_mouth_eyes = cp.bitwise_or(eyes, brow_mouth)
        brows_mouth_eyes = utils.vertical_shift(brows_mouth_eyes, int(head_offset / 2))

        face = utils.blend_transparent(face, brows_mouth_eyes)  # голова + рот + брови + глаза gpu

        face = utils.blend_transparent(face, self.imgs.get_img("hair"))  # голова + рот + брови + глаза + волосы
        face = cv.warpAffine(face.get(), rot, self.imgs.w_s(), flags=linear, borderMode=bmode)

        body = utils.blend_transparent(body, cp.array(face))

        body_shift = cp.array([[1, 0, 0], [0, 1, head_offset + self.cur_breathe]], dtype=cp.float32)
        body_rot = utils.get_rot_mat((BODY_ROT_X, BODY_ROT_Y), self.head_tilt / 4)
        body_rot = cp.vstack([body_rot, [0, 0, 1]])
        body_rot = cp.matmul(body_shift, body_rot)
        body = cv.warpAffine(body.get(), body_rot.get(), self.imgs.w_s(), flags=linear, borderMode=bmode)

        self.res = utils.blend_transparent(self.imgs.get_img("background"), cp.array(body))

    def display(self):
        cv.imshow("Animezator", self.res.get())

    def change_overlay(self, overlay_id):
        self.imgs.change_overlay(overlay_id)

    def toggle_animation(self, animation_id):
        self.imgs.toggle_animation(animation_id)

    def update_animations(self):
        self.imgs.update_animation()


def make_eyes(l_pupil_pos, r_pupil_pos, imgs, l_eye_shape, r_eye_shape):
    l_eye_pupil = imgs.get_img("l_eye_pupil").copy()
    l_eye_pupil = utils.horizontal_shift(l_eye_pupil, int(l_pupil_pos))
    r_eye_pupil = imgs.get_img("r_eye_pupil").copy()
    r_eye_pupil = utils.horizontal_shift(r_eye_pupil, int(r_pupil_pos))
    l_eye_white = imgs.get_img("l_eye_white_" + str(l_eye_shape))
    r_eye_white = imgs.get_img("r_eye_white_" + str(r_eye_shape))
    eye_white = cp.bitwise_or(l_eye_white, r_eye_white)
    eye_pupil = cp.bitwise_or(l_eye_pupil, r_eye_pupil)
    eye_pupil[eye_white[:, :, 3] == 0] = 0
    res_eye = utils.blend_transparent(eye_white, eye_pupil)
    l_eye_s = imgs.get_img("l_eye_" + str(l_eye_shape))
    r_eye_s = imgs.get_img("r_eye_" + str(r_eye_shape))
    eye_shape = cp.bitwise_or(l_eye_s, r_eye_s)
    res_eye = utils.blend_transparent(res_eye, eye_shape)
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


def make_brow_warp_matrix(y_offset, rot_x, rot_y, angle):
    brow_shift = cp.array([[1, 0, 0], [0, 1, -y_offset]], dtype=cp.float32)
    brow_rot = utils.get_rot_mat((rot_x, rot_y), angle)
    brow_rot = cp.vstack([brow_rot, [0, 0, 1]])
    brow_rot = cp.matmul(brow_shift, brow_rot)
    return brow_rot.get()


def breathe(cur, status):
    if status == 1:
        if abs(cur - BREATHING_Y_OFFSET) > 0.01:
            cur += BREATHING_SPD
        else:
            status = -1
            cur -= BREATHING_SPD
    else:
        if abs(cur + BREATHING_Y_OFFSET) > 0.01:
            cur -= BREATHING_SPD
        else:
            status = 1
            cur += BREATHING_SPD
    return cur, status
