import utils


class Overlay:
    has_background = 0
    has_hair_back = 0
    has_head = 0
    has_l_brow = 0
    has_r_brow = 0
    has_mouth = 0
    has_r_eye = 0
    has_l_eye = 0
    has_hair = 0

    def __init__(self, overlay):
        replaces = overlay["replaces"]
        self.has_background, self.background = load_image("background", replaces)
        self.has_hair_back, self.hair_back = load_image("hair_back", replaces)
        self.has_head, self.head = load_image("head", replaces)
        self.has_l_brow, self.l_brow = load_image("l_brow", replaces)
        self.has_r_brow, self.r_brow = load_image("r_brow", replaces)
        self.has_mouth, self.mouth = load_image("mouth", replaces)
        self.has_r_eye, self.r_eye = load_image("r_eye", replaces)
        self.has_l_eye, self.l_eye = load_image("l_eye", replaces)
        self.has_hair, self.hair = load_image("hair", replaces)


def load_image(name, replaces):
    if name in replaces:
        if replaces[name] != "":
            return 1, utils.load_image(replaces[name])  # загружена картинка
        else:
            return -1, None  # картинка не используется
    else:
        return 0, None  # картинка не изменяется
