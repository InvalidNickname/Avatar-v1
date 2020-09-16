from imutils import face_utils
import imutils
import dlib
import cv2 as cv
import math
import numpy as np
import animator as anim
import utils


def get_mouth_shape(upper_point, lower_point, rel_h):
    dst = utils.length(upper_point, lower_point) / rel_h
    mouth_shape = 3
    if dst < 0.1:
        mouth_shape = 0
    elif dst < 0.2:
        mouth_shape = 1
    elif dst < 0.3:
        mouth_shape = 2
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


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/model.dat")

    animator = anim.Animator()

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
        if rects:
            # определение точек лица
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            # отрисовка лица и точек на оригинале
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            for (x, y) in shape:
                cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
            # начинаем изврат
            # все размеры определяем относительно длины носа, т.к. она неизменна
            rel_h = shape[33][1] - shape[27][1]
            # текушее увеличение лица
            # определяем поворот лица по двум точкам (35 и 45) - дальний край левого глаза и левый край носа
            a = shape[45][0] - shape[35][0]
            b = utils.length(shape[45], shape[35])
            alpha = math.degrees(math.acos(a / b) - 1)
            # определяем положение рта, точки 62 и 66
            mouth_shape = get_mouth_shape(shape[62], shape[66], rel_h)
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
            left_brow_pos_delta = (utils.length(shape[35], shape[24]) / rel_h - 1.5) * 10
            right_brow_pos_delta = (utils.length(shape[35], shape[19]) / rel_h - 1.5) * 10
            # отрисовка и отображение
            animator.animate(alpha, left_brow_pos_delta * 2, right_brow_pos_delta * 2, r_pupil_pos, l_pupil_pos)
            animator.put_mask(mouth_shape, right_eye_shape, left_eye_shape)
            animator.display()
        else:
            cv.putText(frame, "Face not found", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 32, 32))

        cv.imshow("Output", frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
