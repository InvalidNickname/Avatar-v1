######################
# основные настройки #
######################

# показывать ли вид с камеры
SHOW_INPUT = True

# показывать ли отрисованную картинку
SHOW_OUTPUT = True

# ограничение fps
TARGET_FPS = 24

# цвет фона (для хромакея, например), BGRA
B_COLOR = (0, 255, 0, 255)

# во сколько раз уменьшать исходные картинки для анимации
DOWNSCALING = 1

############################
# ограничения при анимации #
############################

# максимальный угол наклона головы на картинке
LIMIT_HEAD_TILT = 15

# насколько высоко можно поднять/опустить брови на картинке
LIMIT_BROW_HIGH = int(10 / DOWNSCALING)
LIMIT_BROW_LOW = int(-2 / DOWNSCALING)

# максимальное смещение головы по вертикали/горизонтали при наклоне
HEAD_MAX_Y_TILT = int(8 / DOWNSCALING)
HEAD_MAX_X_TILT = int(16 / DOWNSCALING)

###########################
# ключевые точки анимации #
###########################

# точка вращения головы относительно левого верхнего края картинки
HEAD_ROT_POINT_Y = int(404 / DOWNSCALING)
HEAD_ROT_POINT_X = int(274 / DOWNSCALING)

# y точки вращения неподвижной части волос
UM_HAIR_ROT_POINT_Y = int(300 / DOWNSCALING)

# точка вращения правой брови относительно левого верхнего края картинки
R_BROW_ROT_X = int(194 / DOWNSCALING)
R_BROW_ROT_Y = int(262 / DOWNSCALING)

# точка вращения левой брови относительно левого верхнего края картинки
L_BROW_ROT_X = int(314 / DOWNSCALING)
L_BROW_ROT_Y = int(256 / DOWNSCALING)

# точка вращения тела относительно левого верхнего края картинки
BODY_ROT_Y = int(548 / DOWNSCALING)
BODY_ROT_X = int(264 / DOWNSCALING)

# сдвиг тела по вертикали при "дыхании"
BREATHING_Y_OFFSET = int(6 / DOWNSCALING)
# скорость сдвига тела по вертикали при "дыхании"
BREATHING_SPD = 0.4 / DOWNSCALING

###################################################
# контуры изображений для более быстрой отрисовки #
###################################################

# левая бровь, указываются левая верхняя и нижняя правая точки потенциального пространства отрисовки
L_BROW_BB = (int(264 / DOWNSCALING), int(137 / DOWNSCALING), int(396 / DOWNSCALING), int(274 / DOWNSCALING))

# правая бровь
R_BROW_BB = (int(132 / DOWNSCALING), int(137 / DOWNSCALING), int(264 / DOWNSCALING), int(274 / DOWNSCALING))
