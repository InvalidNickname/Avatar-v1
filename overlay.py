import cupy as cp

import utils


def get_part_name_for_loading(part):
    if part == "mouth":
        return ["mouth_0", "mouth_1", "mouth_2", "mouth_3", "mouth_4", "mouth_5", "mouth_6", "mouth_7", "mouth_8",
                "mouth_9", "mouth_10", "mouth_11", "mouth_12", "mouth_13"]
    elif part == "r_eye_white":
        return ["r_eye_white_0", "r_eye_white_1", "r_eye_white_2", "r_eye_white_3", "r_eye_white_4", "r_eye_white_5",
                "r_eye_white_6"]
    elif part == "l_eye_white":
        return ["l_eye_white_0", "l_eye_white_1", "l_eye_white_2", "l_eye_white_3", "l_eye_white_4", "l_eye_white_5",
                "l_eye_white_6"]
    elif part == "r_eye":
        return ["r_eye_0", "r_eye_1", "r_eye_2", "r_eye_3", "r_eye_4", "r_eye_5", "r_eye_6"]
    elif part == "l_eye":
        return ["l_eye_0", "l_eye_1", "l_eye_2", "l_eye_3", "l_eye_4", "l_eye_5", "l_eye_6"]
    else:
        return [part]


class Overlay:
    overlay_id = 0
    animation_started = False
    animation_id = -1
    animation_step = -1
    animation_status = 0
    speed_counter = -1

    def __init__(self, json_overlays, json_animations):
        self.overlays = []
        for i in range(len(json_overlays)):
            replaces = json_overlays[i]["replaces"]
            self.overlays.append(dict())
            for j in replaces:
                names = get_part_name_for_loading(j)
                for part in names:
                    self.overlays[i][part] = self.load_image(replaces[j])
        self.animations = []
        for i in range(len(json_animations)):
            self.animations.append(dict())
            for j in json_animations[i]:
                self.animations[i][j] = json_animations[i][j]
            frames = json_animations[i]["frames"]
            self.animations[i]["frames"] = []
            for j in range(len(frames)):
                self.animations[i]["frames"].append(self.load_image(frames[j]))

    def get_img(self, part):
        if self.animation_id != -1 and part == self.animations[self.animation_id]["replaces"]:
            return self.animations[self.animation_id]["frames"][self.animation_step]
        if part in self.overlays[self.overlay_id]:
            return self.overlays[self.overlay_id][part]
        else:
            return self.overlays[0][part]

    def shape(self):
        return self.overlays[0]["background"].shape

    def w_s(self):
        new_shape = (self.overlays[0]["background"].shape[1], self.overlays[0]["background"].shape[0])
        return new_shape

    def load_image(self, path):
        if path != "":
            return utils.load_image(path)
        else:
            return cp.zeros(self.overlays[0]["background"].shape, dtype=cp.uint8)

    def toggle_animation(self, anim_id):
        overlay_str = '|' + str(self.overlay_id) + '|'
        if overlay_str in self.animations[self.animation_id]["for_overlays"]:
            if anim_id != self.animation_id:
                self.animation_status = 1
            self.animation_id = anim_id
            if self.animation_status == -1:
                self.animation_step = len(self.animations[self.animation_id]["frames"]) - 1
            else:
                self.animation_step = 0
            self.animation_started = True
            self.speed_counter = -1

    def change_overlay(self, over_id):
        self.overlay_id = over_id
        self.animation_id = -1
        self.animation_started = False

    def update_animation(self):
        if self.animation_started:
            self.speed_counter += 1
            if self.speed_counter == self.animations[self.animation_id]["speed"]:
                self.speed_counter = 0
                if self.animation_status == 1:
                    if self.animation_step < len(self.animations[self.animation_id]["frames"]) - 1:
                        self.animation_step += 1
                    else:
                        self.animation_started = False
                        if self.animations[self.animation_id]["on_end"] != "leave":
                            self.animation_id = -1  # прекратить показ анимации
                        if self.animations[self.animation_id]["on_repeat"] == "reverse":
                            self.animation_status = -1  # проигрывать в обратном порядке при следующем запуске
                elif self.animation_status == -1:
                    if self.animation_step > 0:
                        self.animation_step -= 1
                    else:
                        self.animation_started = False
                        self.animation_id = -1  # прекратить показ анимации
                        if self.animations[self.animation_id]["on_repeat"] == "reverse":
                            self.animation_status = 1  # проигрывать в нормальном порядке при следующем запуске
