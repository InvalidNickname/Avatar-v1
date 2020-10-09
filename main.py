from imutils import face_utils
import imutils
import dlib
import cv2 as cv
import math
import numpy as np
import time
import random

import animator as anim
import utils
import json


def get_mouth_shape(upper_point, lower_point, rel_h, mean_mouth, corners, face_rot):
    dst = utils.length(upper_point, lower_point) / rel_h
    # угол поворота уголков рта относительно лица
    alpha = math.degrees(math.atan((corners[2][1] - corners[3][1]) / (corners[3][0] - corners[2][0]))) - face_rot
    if dst < 0.1:
        if alpha > 15:
            mouth_shape = 6
        else:
            mouth_shape = 0
    elif dst < 0.2:
        if alpha > 15:
            mouth_shape = 7
        else:
            mouth_shape = 1
    elif dst < 0.3:
        if mean_mouth < 200:
            if alpha > 15:
                mouth_shape = 8
            else:
                mouth_shape = 2
        else:
            # зубы показаны полностью
            mouth_shape = 4
    else:
        if mean_mouth < 200:
            if alpha > 15:
                mouth_shape = 9
            else:
                mouth_shape = 3
        else:
            # зубы показаны полностью
            mouth_shape = 5
    return mouth_shape


def get_eye_shape(upper_point, lower_point, rel_h):
    dst = utils.length(upper_point, lower_point) / rel_h
    eye_shape = 3
    if dst < 0.09:
        eye_shape = 0
    elif dst < 0.12:
        eye_shape = 1
    elif dst < 0.14:
        eye_shape = 2
    return eye_shape


def get_pupil_pos(eye_area, right_point, left_point):
    eye_area = cv.equalizeHist(eye_area)
    _, thresh_gray = cv.threshold(eye_area, 0, 255, cv.THRESH_BINARY)
    max_cols = np.amin(thresh_gray, axis=0)
    occurrences = np.where(max_cols == 0)[0]
    max_index = (occurrences[len(occurrences) - 1] + occurrences[0]) / 2
    pupil_pos = max_index - (right_point[0] + left_point[0]) / 2 + left_point[0]
    return pupil_pos


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


def main():
    with open("data/overlays.json", "r") as read_file:
        overlays = json.load(read_file)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/model.dat")

    animator = anim.Animator(overlays)

    next_blink = random.random() * 2 + 3
    prev_blink = time.time()

    standby_counter = 0

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.flip(frame, 1)
        frame = imutils.resize(frame, width=500)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        # обнаружение лица
        rects = detector(gray, 1)
        head_center = frame.shape[0] / 2
        if rects:
            # сброс счётчика автономного режима
            standby_counter = 0
            # определение точек лица
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            # если не все лицо в кадре - скип
            if shape.shape[0] != 68:
                continue
            # отрисовка лица и точек на оригинале
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            draw_landmarks(frame, shape)
            # начинаем изврат
            if animator.head_central_y == -1:
                animator.head_central_y = frame.shape[0]
            # все размеры определяем относительно длины носа, т.к. она неизменна
            rel_h = shape[33][1] - shape[27][1]
            # текушее увеличение лица
            # определяем поворот лица по двум точкам (37 и 46) - дальние края глаз
            a = shape[45][0] - shape[36][0]
            b = shape[36][1] - shape[45][1]
            alpha = math.degrees(math.atan(b / a))
            # определяем положение рта, точки 62 и 66
            mouth_region = gray[shape[62][1]:shape[66][1], shape[61][0]:shape[63][0]].copy()
            cv.threshold(mouth_region, 80, 256, cv.THRESH_BINARY, mouth_region)
            mean_mouth = cv.mean(mouth_region)
            mouth_corners = (shape[48], shape[60], shape[64], shape[54])
            mouth_shape = get_mouth_shape(shape[62], shape[66], rel_h, mean_mouth[0], mouth_corners, alpha)
            # определяем положение глаз
            # правый глаз - точки 37, 41
            right_eye_shape = get_eye_shape(shape[37], shape[41], rel_h)
            right_eye_area = gray[shape[20][1]:shape[30][1], shape[36][0]:shape[39][0]]
            r_pupil_pos = get_pupil_pos(right_eye_area, shape[39], shape[36])
            # левый глаз - точки 43, 47
            left_eye_shape = get_eye_shape(shape[44], shape[46], rel_h)
            left_eye_area = gray[shape[23][1]:shape[30][1], shape[42][0]:shape[44][0]]
            l_pupil_pos = get_pupil_pos(left_eye_area, shape[45], shape[42])
            # брови
            left_brow_pos_delta = (utils.length(shape[27], shape[21]) / rel_h - 0.3) * 10
            right_brow_pos_delta = (utils.length(shape[27], shape[22]) / rel_h - 0.3) * 10
            # сдвиг головы по вертикали
            head_center = shape[33][1]
            # отрисовка
            animator.animate(alpha, left_brow_pos_delta * 2, right_brow_pos_delta * 2, r_pupil_pos, l_pupil_pos,
                             shape[33][1])
            # проверка моргания
            prev_blink, next_blink = check_blinking(prev_blink, next_blink, animator)
            animator.put_mask(mouth_shape, right_eye_shape, left_eye_shape)
            # отображение
            animator.display()
        else:
            if standby_counter == 0:
                standby_counter = time.time()
            cv.putText(frame, "Face not found", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 32, 32))
            prev_blink, next_blink = check_blinking(prev_blink, next_blink, animator)
            # если лицо не найдено через 3 секунды - анимация автономного режима, иначе просто моргаем
            animator.standby(time.time() - standby_counter > 3)

        cv.imshow("Output", frame)
        key = cv.waitKey(1)
        if key == ord('q') or key == ord('é'):
            break
        elif key == ord('r') or key == ord('ê'):
            animator.head_central_y = head_center
        else:
            for overlay_key in overlays:
                if key == ord(overlay_key["key"]):
                    animator.change_overlay(overlay_key["id"])

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
