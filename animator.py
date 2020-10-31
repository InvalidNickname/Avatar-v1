import cv2 as cv
import random
import numpy as np

from overlay import *
from limits import *

bmode = cv.BORDER_REPLICATE
linear = cv.INTER_LINEAR


class Animator:
    # текущие углы поворота головы
    cur_tilt = 0
    cur_ver_tilt = 0
    cur_hor_tilt = 0
    # текущее смещение головы от наклона
    cur_hor_offset = 0
    cur_ver_offset = 0
    # текущая высота бровей
    cur_l_brow = 0
    cur_r_brow = 0
    # текущий наклон бровей
    cur_r_brow_tilt = 0
    cur_l_brow_tilt = 0
    # текущее расположение зрачков
    c_r_pupil = 0
    c_l_pupil = 0
    # текущая форма глаз
    c_r_eye_s = 6
    c_l_eye_s = 6

    # моргание
    blinking = 0  # 0 - нет, -1 - закрывает глаза, 1 - открывает глаза
    blinking_mat = [0, 2, 3, 5]  # изображения глаз при моргании
    blinking_step = 4

    head_tilt = 0
    target_head_tilt = 0
    res = None

    cur_breathe = 0
    breathe_status = 1

    def __init__(self, overlays, animations, limits):
        self.imgs = Overlay(overlays, animations, limits)

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
                    if self.c_r_eye_s < self.blinking_mat[len(self.blinking_mat) - i - 1]:
                        self.blinking_step = len(self.blinking_mat) - i - 1
            if self.blinking_step > 0:
                self.blinking_step -= 1
            else:
                self.blinking = 1
                self.blinking_step += 1
        self.c_r_eye_s = self.blinking_mat[self.blinking_step]
        self.c_l_eye_s = self.c_r_eye_s

    def animate(self, angle, l_brow_pos, r_brow_pos, r_pupil_pos, l_pupil_pos, l_brow_tilt,
                r_brow_tilt, head_ver_tilt, head_horizontal_tilt):
        # поворачиваем лицо
        self.cur_tilt = move_slowly(angle, self.cur_tilt, 3)
        self.cur_tilt = set_limits_2(self.cur_tilt, "head", "rot", self.imgs.limits)
        # поворачиваем голову
        if head_ver_tilt < 0:
            head_ver_tilt = -head_ver_tilt - 180
        else:
            head_ver_tilt = -head_ver_tilt + 180
        self.cur_ver_tilt = move_slowly(head_ver_tilt, self.cur_ver_tilt, 5)
        self.cur_ver_tilt = set_limits_2(self.cur_ver_tilt, "head", "rot", self.imgs.limits)
        self.cur_hor_tilt = move_slowly(head_horizontal_tilt, self.cur_hor_tilt, 5)
        self.cur_hor_tilt = set_limits_2(self.cur_hor_tilt, "head", "rot", self.imgs.limits)
        self.cur_hor_offset = self.cur_hor_tilt / self.imgs.lim("head", "rot_max") * self.imgs.lim("head", "x_max")
        self.cur_ver_offset = self.cur_ver_tilt / self.imgs.lim("head", "rot_max") * self.imgs.lim("head", "y_max")
        # двигаем брови
        self.cur_l_brow = move_slowly(l_brow_pos, self.cur_l_brow, 2)
        self.cur_l_brow = set_limits_2(self.cur_l_brow, "l_brow", "y", self.imgs.limits)
        self.cur_l_brow_tilt = move_slowly(self.cur_l_brow_tilt, l_brow_tilt, 2)
        self.cur_r_brow = move_slowly(r_brow_pos, self.cur_r_brow, 2)
        self.cur_r_brow = set_limits_2(self.cur_r_brow, "r_brow", "y", self.imgs.limits)
        self.cur_r_brow_tilt = move_slowly(self.cur_r_brow_tilt, r_brow_tilt, 2)
        # двигаем зрачки
        self.c_r_pupil = move_slowly(r_pupil_pos, self.c_r_pupil, 2)
        self.c_l_pupil = move_slowly(l_pupil_pos, self.c_l_pupil, 2)
        # дыхание
        self.cur_breathe, self.breathe_status = breathe(self.cur_breathe, self.breathe_status)

    def standby_animate(self):
        if abs(self.target_head_tilt - self.head_tilt) < 0.05:
            self.target_head_tilt = random.random() * 20 - 10
        self.head_tilt = move_slowly(self.target_head_tilt, self.head_tilt, 7)
        self.animate(self.head_tilt, 0, 0, 0, 0, 0, 0, 0, 0)

    def standby(self, animate):
        if animate:
            self.standby_animate()
        if self.blinking == 0:
            self.c_r_eye_s = 6
            self.c_l_eye_s = 6
        else:
            self.blink()
        self.put_mask(0, 6, 6)
        self.display()

    def put_mask(self, mouth_shape, r_eye_s, l_eye_s):
        hair_back_rot = utils.get_rot_mat(self.imgs.rp_with_bb("hair_back"), self.cur_tilt, True)
        hair_back = self.imgs.get_img("hair_back").get()
        body = utils.warp_with_bb(hair_back, self.imgs.get_bb("hair_back"), hair_back_rot)
        body = utils.blend_with_bb(cp.array(body), self.imgs.get_img("body"), self.imgs.get_bb("body"))

        l_brow_m = make_brow_warp_matrix(self.cur_l_brow, self.imgs.rp_with_bb("l_brow"), self.cur_l_brow_tilt)
        l_brow = self.imgs.get_img("l_brow").get()
        l_brow = utils.warp_with_bb(l_brow, self.imgs.get_bb("l_brow"), l_brow_m)

        r_brow_m = make_brow_warp_matrix(self.cur_r_brow, self.imgs.rp_with_bb("r_brow"), self.cur_r_brow_tilt)
        r_brow = self.imgs.get_img("r_brow").get()
        r_brow = utils.warp_with_bb(r_brow, self.imgs.get_bb("r_brow"), r_brow_m)

        s_face = utils.bitwise_or_with_bb(cp.array(r_brow), cp.array(l_brow), self.imgs.get_bb("l_brow"))
        mouth_st = "mouth_" + str(mouth_shape)
        s_face = utils.bitwise_or_with_bb(s_face, self.imgs.get_img(mouth_st), self.imgs.get_bb(mouth_st))

        if self.blinking == 0:
            self.c_r_eye_s = r_eye_s
            self.c_l_eye_s = l_eye_s
        else:
            self.blink()

        eyes_bb = utils.combine_bbs(self.imgs.get_bb("l_eye_" + str(self.c_l_eye_s)),
                                    self.imgs.get_bb("r_eye_" + str(self.c_r_eye_s)))
        eyes = make_eyes(self.c_l_pupil, self.c_r_pupil, self.imgs, self.c_l_eye_s, self.c_r_eye_s, eyes_bb)

        s_face = utils.bitwise_or_with_bb(s_face, eyes, eyes_bb)
        s_face = utils.shift(s_face, -self.cur_ver_offset / 1.5, -self.cur_hor_offset / 2.5)

        hair_shift = utils.get_shift_mat(-self.cur_hor_offset / 5, -self.cur_ver_offset / 3)
        un_rot = utils.get_rot_mat(self.imgs.rp_with_bb("hair_um"), -self.cur_tilt / 2, True)
        un_rot = np.vstack([un_rot, [0, 0, 1]])
        un_rot = np.matmul(hair_shift, un_rot)
        hair_um = utils.warp_with_bb(self.imgs.get_img("hair_um").get(), self.imgs.get_bb("hair_um"), un_rot)

        s_face = utils.bitwise_or_with_bb(s_face, cp.array(hair_um), self.imgs.get_bb("hair_um"))
        face = utils.blend_with_bb(self.imgs.get_img("head"), s_face, self.imgs.get_bb("head"))

        hair = utils.shift(self.imgs.get_img("hair"), -self.cur_ver_offset / 3, -self.cur_hor_offset / 5)
        face = utils.blend_with_bb(face, hair, self.imgs.get_bb("hair"))
        face = utils.shift(face, -self.cur_ver_offset, -self.cur_hor_offset)
        rot = utils.get_rot_mat(self.imgs.rp_with_bb("head"), self.cur_tilt, True)
        face = utils.warp_with_bb(face.get(), self.imgs.get_bb("head"), rot)

        face_bb = utils.combine_bbs(self.imgs.get_bb("head"), self.imgs.get_bb("hair"))
        body = utils.blend_with_bb(body, cp.array(face), face_bb)

        body_shift = utils.get_shift_mat(0, self.cur_breathe)
        body_rot = utils.get_rot_mat(self.imgs.rp("body"), self.cur_tilt / 4, True)
        body_rot = np.vstack([body_rot, [0, 0, 1]])
        body_rot = np.matmul(body_shift, body_rot)
        body = cv.warpAffine(body.get(), body_rot, self.imgs.w_s(), flags=linear, borderMode=bmode)

        self.res = cp.zeros(self.imgs.shape(), dtype=cp.uint8)
        self.res[:, :, 0] = B_COLOR[0]
        self.res[:, :, 1] = B_COLOR[1]
        self.res[:, :, 2] = B_COLOR[2]
        self.res[:, :, 3] = B_COLOR[3]
        self.res = utils.blend_with_bb(self.res, self.imgs.get_img("background"), self.imgs.get_bb("background"))
        body_bb = utils.combine_bbs(face_bb, self.imgs.get_bb("body"))
        body_bb = utils.combine_bbs(body_bb, self.imgs.get_bb("hair_back"))
        self.res = utils.blend_with_bb(self.res, cp.array(body), body_bb)

    def display(self):
        cv.imshow("Animezator", self.res.get())

    def change_overlay(self, overlay_id):
        self.imgs.change_overlay(overlay_id)

    def toggle_animation(self, animation_id):
        self.imgs.toggle_animation(animation_id)

    def update_animations(self):
        self.imgs.update_animation()

    def get_overlay(self):
        return self.imgs.overlay_id

    def get_cur_animations(self):
        return self.imgs.get_cur_animations()


def make_eyes(l_pupil_pos, r_pupil_pos, imgs, l_eye_shape, r_eye_shape, bb):
    l_eye_pupil = imgs.get_img("l_eye_pupil").copy()
    l_eye_pupil = utils.horizontal_shift(l_eye_pupil, int(l_pupil_pos))
    r_eye_pupil = imgs.get_img("r_eye_pupil").copy()
    r_eye_pupil = utils.horizontal_shift(r_eye_pupil, int(r_pupil_pos))
    l_eye_white = imgs.get_img("l_eye_white_" + str(l_eye_shape))
    r_eye_white = imgs.get_img("r_eye_white_" + str(r_eye_shape))
    eye_white = utils.bitwise_or_with_bb(l_eye_white, r_eye_white, bb)
    eye_pupil = utils.bitwise_or_with_bb(l_eye_pupil, r_eye_pupil, bb)
    eye_pupil[eye_white[:, :, 3] == 0] = 0
    res_eye = utils.blend_with_bb(eye_white, eye_pupil, bb)
    l_eye_s = imgs.get_img("l_eye_" + str(l_eye_shape))
    r_eye_s = imgs.get_img("r_eye_" + str(r_eye_shape))
    eye_shape = utils.bitwise_or_with_bb(l_eye_s, r_eye_s, bb)
    res_eye = utils.blend_with_bb(res_eye, eye_shape, bb)
    return res_eye


def move_slowly(direction, current, multiplier):
    if direction < current:
        current -= (current - direction) / multiplier
    elif direction > current:
        current += (direction - current) / multiplier
    return current


def set_limits_2(current, part, limit, limits):
    if current > limits[part][limit + "_max"]:
        current = limits[part][limit + "_max"]
    elif current < limits[part][limit + "_min"]:
        current = limits[part][limit + "_min"]
    return current


def set_limits(current, upper_limit, lower_limit):
    if current > upper_limit:
        current = upper_limit
    elif current < lower_limit:
        current = lower_limit
    return current


def make_brow_warp_matrix(y_offset, rot, angle):
    brow_shift = utils.get_shift_mat(0, -y_offset)
    brow_rot = utils.get_rot_mat(rot, angle, True)
    brow_rot = np.vstack([brow_rot, [0, 0, 1]])
    brow_rot = np.matmul(brow_shift, brow_rot)
    return brow_rot


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
