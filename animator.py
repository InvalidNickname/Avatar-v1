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
    cur_vertical_tilt = 0
    cur_horizontal_tilt = 0
    # текущее смещение головы от наклона
    cur_tilt_hor_offset = 0
    cur_tilt_ver_offset = 0
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

    def __init__(self, overlays, animations):
        self.imgs = Overlay(overlays, animations)
        self.l_brow_rot_x = L_BROW_ROT_X - L_BROW_BB[0]
        self.l_brow_rot_y = L_BROW_ROT_Y - L_BROW_BB[1]
        self.r_brow_rot_x = R_BROW_ROT_X - R_BROW_BB[0]
        self.r_brow_rot_y = R_BROW_ROT_Y - R_BROW_BB[1]
        self.l_b_shape = (L_BROW_BB[2] - L_BROW_BB[0], L_BROW_BB[3] - L_BROW_BB[1])
        self.r_b_shape = (R_BROW_BB[2] - R_BROW_BB[0], R_BROW_BB[3] - R_BROW_BB[1])

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
                r_brow_tilt, head_vertical_tilt, head_horizontal_tilt):
        # поворачиваем лицо
        self.cur_tilt = move_slowly(angle, self.cur_tilt, 3)
        self.cur_tilt = set_limits(self.cur_tilt, LIMIT_HEAD_TILT, -LIMIT_HEAD_TILT)
        # поворачиваем голову
        if head_vertical_tilt < 0:
            head_vertical_tilt = -head_vertical_tilt - 180
        else:
            head_vertical_tilt = -head_vertical_tilt + 180
        self.cur_vertical_tilt = move_slowly(head_vertical_tilt, self.cur_vertical_tilt, 5)
        self.cur_vertical_tilt = set_limits(self.cur_vertical_tilt, LIMIT_HEAD_TILT, -LIMIT_HEAD_TILT)
        self.cur_horizontal_tilt = move_slowly(head_horizontal_tilt, self.cur_horizontal_tilt, 5)
        self.cur_horizontal_tilt = set_limits(self.cur_horizontal_tilt, LIMIT_HEAD_TILT, -LIMIT_HEAD_TILT)
        self.cur_tilt_hor_offset = self.cur_horizontal_tilt / LIMIT_HEAD_TILT * HEAD_MAX_X_TILT
        self.cur_tilt_ver_offset = self.cur_vertical_tilt / LIMIT_HEAD_TILT * HEAD_MAX_Y_TILT
        # двигаем брови
        self.cur_l_brow = move_slowly(l_brow_pos, self.cur_l_brow, 2)
        self.cur_l_brow = set_limits(self.cur_l_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
        self.cur_l_brow_tilt = move_slowly(self.cur_l_brow_tilt, l_brow_tilt, 2)
        self.cur_r_brow = move_slowly(r_brow_pos, self.cur_r_brow, 2)
        self.cur_r_brow = set_limits(self.cur_r_brow, LIMIT_BROW_HIGH, LIMIT_BROW_LOW)
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
        rot = utils.get_rot_mat((HEAD_ROT_POINT_X, HEAD_ROT_POINT_Y), self.cur_tilt, True)

        hair_back = self.imgs.get_img("hair_back").get()
        body = cv.warpAffine(hair_back, rot, self.imgs.w_s(), flags=linear, borderMode=bmode)
        body = utils.blend_transparent(cp.array(body), self.imgs.get_img("body"))

        l_brow_m = make_brow_warp_matrix(self.cur_l_brow, self.l_brow_rot_x, self.l_brow_rot_y, self.cur_l_brow_tilt)
        l_brow = self.imgs.get_img("l_brow").get()
        l_brow_part = l_brow[L_BROW_BB[1]:L_BROW_BB[3], L_BROW_BB[0]:L_BROW_BB[2], :]
        l_brow_part = cv.warpAffine(l_brow_part, l_brow_m, self.l_b_shape, flags=linear, borderMode=bmode)
        l_brow[L_BROW_BB[1]:L_BROW_BB[3], L_BROW_BB[0]:L_BROW_BB[2], :] = l_brow_part

        r_brow_m = make_brow_warp_matrix(self.cur_r_brow, self.r_brow_rot_x, self.r_brow_rot_y, self.cur_r_brow_tilt)
        r_brow = self.imgs.get_img("r_brow").get()
        r_brow_part = r_brow[R_BROW_BB[1]:R_BROW_BB[3], R_BROW_BB[0]:R_BROW_BB[2], :]
        r_brow_part = cv.warpAffine(r_brow_part, r_brow_m, self.r_b_shape, flags=linear, borderMode=bmode)
        r_brow[R_BROW_BB[1]:R_BROW_BB[3], R_BROW_BB[0]:R_BROW_BB[2], :] = r_brow_part

        s_face = cp.bitwise_or(cp.array(r_brow), cp.array(l_brow))
        mouth = make_mouth(self.imgs, mouth_shape, self.cur_tilt_hor_offset / 3, self.cur_tilt_ver_offset / 3)
        s_face = cp.bitwise_or(s_face, mouth)

        if self.blinking == 0:
            self.c_r_eye_s = r_eye_s
            self.c_l_eye_s = l_eye_s
        else:
            self.blink()

        eyes = make_eyes(self.c_l_pupil, self.c_r_pupil, self.imgs, self.c_l_eye_s, self.c_r_eye_s)

        s_face = cp.bitwise_or(eyes, s_face)
        s_face = utils.shift(s_face, -self.cur_tilt_ver_offset / 1.5, -self.cur_tilt_hor_offset / 2.5)

        hair_shift = utils.get_shift_mat(-self.cur_tilt_hor_offset / 5, -self.cur_tilt_ver_offset / 3)
        un_rot = utils.get_rot_mat((HEAD_ROT_POINT_X, UM_HAIR_ROT_POINT_Y), -self.cur_tilt / 2, True)
        un_rot = np.vstack([un_rot, [0, 0, 1]])
        un_rot = np.matmul(hair_shift, un_rot)
        hair_um = cv.warpAffine(self.imgs.get_img("hair_unmoving").get(), un_rot, self.imgs.w_s(), borderMode=bmode)

        s_face = cp.bitwise_or(s_face, cp.array(hair_um))
        face = utils.blend_transparent(self.imgs.get_img("head"), s_face)

        hair = utils.shift(self.imgs.get_img("hair"), -self.cur_tilt_ver_offset / 3, -self.cur_tilt_hor_offset / 5)
        face = utils.blend_transparent(face, hair)
        face = utils.shift(face, -self.cur_tilt_ver_offset, -self.cur_tilt_hor_offset)
        face = cv.warpAffine(face.get(), rot, self.imgs.w_s(), flags=linear, borderMode=bmode)

        body = utils.blend_transparent(body, cp.array(face))

        body_shift = utils.get_shift_mat(0, self.cur_breathe)
        body_rot = utils.get_rot_mat((BODY_ROT_X, BODY_ROT_Y), self.cur_tilt / 4, True)
        body_rot = np.vstack([body_rot, [0, 0, 1]])
        body_rot = np.matmul(body_shift, body_rot)
        body = cv.warpAffine(body.get(), body_rot, self.imgs.w_s(), flags=linear, borderMode=bmode)

        self.res = cp.zeros(self.imgs.shape(), dtype=cp.uint8)
        self.res[:, :, 0] = B_COLOR[0]
        self.res[:, :, 1] = B_COLOR[1]
        self.res[:, :, 2] = B_COLOR[2]
        self.res[:, :, 3] = B_COLOR[3]
        self.res = utils.blend_transparent(self.res, self.imgs.get_img("background"))
        self.res = utils.blend_transparent(self.res, cp.array(body))

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


def make_mouth(imgs, shape, x_shift, y_shift):
    mouth_contour = imgs.get_img("mouth_" + str(shape))
    mouth_white = imgs.get_img("mouth_white_" + str(shape))
    mouth_base = imgs.get_img("mouth_base").copy()
    mouth_base = utils.shift(mouth_base, y_shift, x_shift)
    mouth_base[mouth_white[:, :, 3] == 0] = 0
    res_mouth = utils.blend_transparent(mouth_base, mouth_contour)
    return res_mouth


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
    brow_shift = utils.get_shift_mat(0, -y_offset)
    brow_rot = utils.get_rot_mat((rot_x, rot_y), angle, True)
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
