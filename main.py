from imutils import face_utils
import imutils
import dlib
import cv2 as cv
import math
import numpy as np
import animator as anim
import utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/model.dat")

animator = anim.Animator()

left_brow_pos_init = -100
right_brow_pos_init = -100
rel_h = -100.

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
        # изначальное увеличение лица
        if rel_h < 0:
            rel_h = shape[33][1] - shape[27][1]
        # текушее увеличение лица
        mod = (shape[33][1] - shape[27][1]) / rel_h
        print(mod)
        # определяем поворот лица по двум точкам (36 и 45) - дальние края глаз
        a = shape[45][0] - shape[35][0]
        b = utils.length(shape[45], shape[35])
        alpha = math.degrees(math.acos(a / b) - 1)
        # определяем положение рта, точки 62 и 66
        dst = utils.length(shape[66], shape[62]) / mod
        mouth_shape = 3
        if dst < 8:
            mouth_shape = 0
        elif dst < 12:
            mouth_shape = 1
        elif dst < 17:
            mouth_shape = 2
        elif dst < 25:
            mouth_shape = 3
        # определяем положение зрачков
        # правый глаз - точки 36, 39
        # левый глаз - точки 42, 45
        # брови
        # инициализация изначального положения бровей
        if left_brow_pos_init < 0:
            left_brow_pos_init = utils.length(shape[44], shape[24]) / mod
            right_brow_pos_init = utils.length(shape[37], shape[19]) / mod
        left_brow_pos_delta = utils.length(shape[44], shape[24]) / mod - left_brow_pos_init
        right_brow_pos_delta = utils.length(shape[37], shape[19]) / mod - right_brow_pos_init
        # отрисовка и отображение
        animator.animate(alpha, left_brow_pos_delta*2, right_brow_pos_delta*2)
        animator.put_mask(mouth_shape)
        animator.display()
    else:
        cv.putText(frame, "Face not found", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 32, 32))

    cv.imshow("Output", frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
