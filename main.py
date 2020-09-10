from imutils import face_utils
import imutils
import dlib
import cv2 as cv
import math
import numpy as np
import animator as anim

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
        # определяем поворот лица по двум точкам (36 и 45) - дальние края глаз
        a = shape[45][0] - shape[35][0]
        b = math.sqrt(a ** 2 + (shape[45][1] - shape[35][1]) ** 2)
        alpha = math.degrees(math.acos(a / b) - 1)
        # отрисовка и отображение
        animator.animate(alpha)
        animator.put_mask()
        animator.display()
    else:
        cv.putText(frame, "Face not found", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 32, 32))

    cv.imshow("Output", frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
