from imutils import face_utils
import imutils
import dlib
import cv2 as cv
import math
import numpy as np
import time
import random
import keyboard
import cupy as cp

import animator as anim
import utils
import json
from limits import *


def get_mouth_shape(upper_point, lower_point, rel_h, mean_mouth, corners, face_rot, right_side):
    dst = utils.length(upper_point, lower_point) / rel_h
    # угол поворота уголков рта относительно лица
    alpha = math.degrees(math.atan((corners[2][1] - corners[3][1]) / (corners[3][0] - corners[2][0]))) - face_rot
    # прямая, соединяющая две правых точки рта
    A = right_side[0][1] - right_side[1][1]
    B = right_side[1][0] - right_side[0][0]
    C = right_side[0][0] * right_side[1][1] - right_side[1][0] * right_side[0][1]
    # расстояние от уголка рта до прямой
    dist = (abs(A * corners[3][0] + B * corners[3][1] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))) / rel_h
    if dst < 0.05:
        if dist < 0.18:
            mouth_shape = 14
        elif alpha > 15:
            mouth_shape = 6
        elif alpha < 0:
            mouth_shape = 10
        else:
            mouth_shape = 0
    elif dst < 0.07:
        if dist < 0.18:
            mouth_shape = 15
        elif alpha > 15:
            mouth_shape = 7
        elif alpha < 0:
            mouth_shape = 11
        else:
            mouth_shape = 1
    elif dst < 0.09:
        if dist < 0.18:
            mouth_shape = 16
        elif mean_mouth < 220:
            if alpha > 15:
                mouth_shape = 8
            elif alpha < 0:
                mouth_shape = 12
            else:
                mouth_shape = 2
        else:
            # зубы показаны полностью
            mouth_shape = 4
    else:
        if dist < 0.12:
            mouth_shape = 17
        elif mean_mouth < 220:
            if alpha > 15:
                mouth_shape = 9
            elif alpha < 0:
                mouth_shape = 13
            else:
                mouth_shape = 3
        else:
            # зубы показаны полностью
            mouth_shape = 5
    return mouth_shape


def get_eye_shape(upper_point, lower_point, rel_h):
    dst = utils.length(upper_point, lower_point) / rel_h
    eye_shape = 6
    if dst < 0.08:
        eye_shape = 0
    elif dst < 0.09:
        eye_shape = 1
    elif dst < 0.10:
        eye_shape = 2
    elif dst < 0.11:
        eye_shape = 3
    elif dst < 0.12:
        eye_shape = 4
    elif dst < 0.13:
        eye_shape = 5
    return eye_shape


def get_pupil_pos(eye_area, right_point, left_point):
    eye_area = cv.equalizeHist(eye_area)
    _, thresh_gray = cv.threshold(eye_area, 0, 255, cv.THRESH_BINARY)
    max_cols = np.amin(thresh_gray, axis=0)
    occurrences = np.where(max_cols == 0)[0]
    if len(occurrences) != 0:
        max_index = (occurrences[len(occurrences) - 1] + occurrences[0]) / 2
        pupil_pos = max_index - (right_point[0] + left_point[0]) / 2 + left_point[0]
        dst = occurrences[len(occurrences) - 1] - occurrences[0]
    else:
        pupil_pos = 0
        dst = 0
    return pupil_pos * 2 / DOWNSCALING, dst


def check_blinking(prev_blink, next_blink, animator):
    cur_time = time.time()
    if prev_blink + next_blink < cur_time:
        animator.blink()
        prev_blink = cur_time
        next_blink = random.random() * 2 + 3
    return prev_blink, next_blink


def draw_landmarks(frame, shape):
    for (x, y) in shape:
        cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
    # точки глаз
    cv.circle(frame, (shape[37][0], shape[37][1]), 1, (255, 0, 0), -1)
    cv.circle(frame, (shape[41][0], shape[41][1]), 1, (255, 0, 0), -1)
    cv.circle(frame, (shape[44][0], shape[44][1]), 1, (255, 0, 0), -1)
    cv.circle(frame, (shape[46][0], shape[46][1]), 1, (255, 0, 0), -1)
    # верх и низ рта
    cv.circle(frame, (shape[62][0], shape[62][1]), 1, (255, 255, 0), -1)
    cv.circle(frame, (shape[66][0], shape[66][1]), 1, (255, 255, 0), -1)
    # уголки рта
    cv.circle(frame, (shape[48][0], shape[48][1]), 1, (0, 255, 255), -1)
    cv.circle(frame, (shape[60][0], shape[60][1]), 1, (0, 255, 255), -1)
    cv.circle(frame, (shape[64][0], shape[64][1]), 1, (0, 255, 255), -1)
    cv.circle(frame, (shape[54][0], shape[54][1]), 1, (0, 255, 255), -1)
    # точки бровей
    cv.circle(frame, (shape[27][0], shape[27][1]), 1, (0, 255, 0), -1)
    cv.circle(frame, (shape[21][0], shape[21][1]), 1, (0, 255, 0), -1)
    cv.circle(frame, (shape[22][0], shape[22][1]), 1, (0, 255, 0), -1)
    cv.circle(frame, (shape[23][0], shape[23][1]), 1, (255, 0, 255), -1)
    cv.circle(frame, (shape[25][0], shape[25][1]), 1, (255, 0, 255), -1)
    cv.circle(frame, (shape[18][0], shape[18][1]), 1, (255, 0, 255), -1)
    cv.circle(frame, (shape[20][0], shape[20][1]), 1, (255, 0, 255), -1)


def rescale_rect(rect, num):
    w = rect.right() - rect.left()
    h = rect.bottom() - rect.top()
    left = rect.left() * num
    top = rect.top() * num
    right = left + w * num
    bottom = top + h * num
    return dlib.rectangle(left, top, right, bottom)


def ref_3d_model():
    model_points = [[0.0, 0.0, 0.0],
                    [0.0, -330.0, -65.0],
                    [-225.0, 170.0, -135.0],
                    [225.0, 170.0, -135.0],
                    [-150.0, -150.0, -125.0],
                    [150.0, -150.0, -125.0]]
    return np.array(model_points, dtype=np.float64)


def ref_2d_image_points(shape):
    image_points = [shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]]
    return np.array(image_points, dtype=np.float64)


def camera_matrix(fl, center):
    matrix = [[fl, 1, center[0]], [0, fl, center[1]], [0, 0, 1]]
    return np.array(matrix, dtype=np.float)


def main():
    cp.cuda.Device(0).use()

    with open("data/overlays.json", "r") as read_file:
        overlays = json.load(read_file)
    with open("data/toggleable_animations.json", "r") as read_file:
        animations = json.load(read_file)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/model.dat")

    animator = anim.Animator(overlays, animations)

    next_blink = random.random() * 2 + 3
    prev_blink = time.time()

    standby_counter = 0

    frames_skipped = 0

    rects = None  # обнаруженные лица

    dt_history = []  # время распознавания за последние кадры
    rt_history = []  # время отрисовки за последние кадры
    ft_history = []  # время кадра за последние кадры

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera, exiting")
        exit()
    while True:
        st = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame not received, exiting")
            break
        frame = cv.flip(frame, 1)
        frame = imutils.resize(frame, width=500)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        # обнаружение лица
        ds = time.time()
        frames_skipped += 1
        if frames_skipped == 3:
            small_gray = imutils.resize(frame, width=250)
            rects = detector.run(small_gray, 1, -0.5)[0]
            frames_skipped = 0
        if rects:
            face = rescale_rect(rects[0], 2)
            # сброс счётчика автономного режима
            standby_counter = 0
            # определение точек лица
            tmp_shape = predictor(gray, face)
            if len(dt_history) >= 10:
                dt_history = dt_history[1:9]
            dt_history.append(time.time() - ds)
            shape = face_utils.shape_to_np(tmp_shape)
            # отрисовка лица и точек на оригинале
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            draw_landmarks(frame, shape)
            # начинаем изврат
            # все размеры определяем относительно длины носа, т.к. она неизменна
            rel_h = shape[33][1] - shape[27][1]
            # текушее увеличение лица
            # определяем поворот лица по двум точкам (37 и 46) - дальние края глаз
            a = shape[45][0] - shape[36][0]
            b = shape[36][1] - shape[45][1]
            alpha = math.degrees(math.atan(b / a))
            # определяем положение лица в пространстве
            face_3d_model = ref_3d_model()
            ref_img_points = ref_2d_image_points(shape)
            h, w = gray.shape
            focal_length = w
            cam_mat = camera_matrix(focal_length, (h / 2, w / 2))
            dists = np.zeros((4, 1), dtype=np.float64)
            _, rot_vec, _ = cv.solvePnP(face_3d_model, ref_img_points, cam_mat, dists)
            r_mat, _ = cv.Rodrigues(rot_vec)
            # первый угол - наклон по вертикали, 2 - поворот по горизонтали
            angles, _, _, _, _, _ = cv.RQDecomp3x3(r_mat)
            # определяем положение рта, точки 62 и 66
            mouth_region = gray[shape[62][1]:shape[66][1], shape[61][0]:shape[63][0]].copy()
            cv.threshold(mouth_region, 80, 256, cv.THRESH_BINARY, mouth_region)
            mean_mouth = cv.mean(mouth_region)
            mouth_corners = (shape[48], shape[60], shape[64], shape[54])
            right_side = (shape[53], shape[55])
            mouth_shape = get_mouth_shape(shape[62], shape[66], rel_h, mean_mouth[0], mouth_corners, alpha, right_side)
            # определяем положение глаз
            # правый глаз - точки 38, 42
            right_eye_shape = get_eye_shape(shape[37], shape[41], rel_h)
            right_eye_area = gray[shape[37][1]:shape[41][1], shape[36][0]:shape[39][0]]
            r_pupil_pos, dst = get_pupil_pos(right_eye_area, shape[39], shape[36])
            if right_eye_shape == 1 and dst > (shape[39][0] - shape[36][0]) / 2:
                right_eye_shape = 0
            # левый глаз - точки 43, 47
            left_eye_shape = get_eye_shape(shape[44], shape[46], rel_h)
            left_eye_area = gray[shape[44][1]:shape[46][1], shape[42][0]:shape[44][0]]
            l_pupil_pos, dst = get_pupil_pos(left_eye_area, shape[45], shape[42])
            if left_eye_shape == 1 and dst > (shape[44][0] - shape[42][0]) / 2:
                left_eye_shape = 0
            # брови
            right_brow_pos_delta = (utils.length(shape[37], shape[19]) / rel_h - 0.3) * 40
            left_brow_pos_delta = (utils.length(shape[44], shape[24]) / rel_h - 0.3) * 40
            # наклон бровей
            a = shape[18][0] - shape[20][0]
            b = shape[20][1] - shape[18][1]
            l_brow_tilt = (alpha - math.degrees(math.atan(b / a))) / 2
            a = shape[23][0] - shape[25][0]
            b = shape[25][1] - shape[23][1]
            r_brow_tilt = (alpha - math.degrees(math.atan(b / a))) / 2
            # анимирование подвижных частей
            animator.animate(alpha, left_brow_pos_delta, right_brow_pos_delta, r_pupil_pos, l_pupil_pos,
                             l_brow_tilt, r_brow_tilt, angles[0], angles[1])
            # проверка моргания
            prev_blink, next_blink = check_blinking(prev_blink, next_blink, animator)
            # отрисовка
            rs = time.time()
            animator.put_mask(mouth_shape, right_eye_shape, left_eye_shape)
            if len(rt_history) >= 10:
                rt_history = rt_history[1:9]
            rt_history.append(time.time() - rs)
            # отображение
            if SHOW_OUTPUT:
                animator.display()
            # обновление анимаций
            animator.update_animations()
        else:
            if standby_counter == 0:
                standby_counter = time.time()
            cv.putText(frame, "face not found", (20, 260), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 32))
            prev_blink, next_blink = check_blinking(prev_blink, next_blink, animator)
            # если лицо не найдено через 3 секунды - анимация автономного режима, иначе просто моргаем
            animator.standby(time.time() - standby_counter > 3)

        ft = time.time() - st
        if len(ft_history) >= 10:
            ft_history = ft_history[1:9]
        ft_history.append(ft)
        fps = str(1 / np.mean(ft_history))[0:3]
        cv.putText(frame, fps + " fps", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 32))
        dt = str(np.mean(dt_history))[0:5] + "/" + str(np.mean(rt_history))[0:5] + "/" + str(np.mean(ft_history))[0:5]
        cv.putText(frame, dt, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 32))

        over_st = "cur overlay: " + str(animator.get_overlay())
        cv.putText(frame, over_st, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 32))
        anim_st = "cur animation: " + animator.get_cur_animations()
        cv.putText(frame, anim_st, (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 32))

        if SHOW_INPUT:
            cv.imshow("Output", frame)

        if keyboard.is_pressed('alt+q'):
            break
        else:
            for overlay_key in overlays:
                if keyboard.is_pressed('alt+' + overlay_key["key"]):
                    animator.change_overlay(overlay_key["id"])
                    break
            for animation_key in animations:
                if keyboard.is_pressed('alt+' + animation_key["key"]):
                    animator.toggle_animation(animation_key["id"])
                    break

        wait_time = int(1000 / TARGET_FPS - 1000 * ft) if ft < 1 / TARGET_FPS else 1
        cv.waitKey(1 if wait_time == 0 else wait_time)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
