import cupy as cp

import utils


def get_part_name_for_loading(part):
    if part == "mouth":
        return ["mouth_0", "mouth_1", "mouth_2", "mouth_3", "mouth_4", "mouth_5", "mouth_6", "mouth_7", "mouth_8",
                "mouth_9", "mouth_10", "mouth_11", "mouth_12", "mouth_13", "mouth_14", "mouth_15", "mouth_16",
                "mouth_17", "mouth_base", "mouth_white_0", "mouth_white_1", "mouth_white_2", "mouth_white_3",
                "mouth_white_4", "mouth_white_5", "mouth_white_6", "mouth_white_7", "mouth_white_8", "mouth_white_9",
                "mouth_white_10", "mouth_white_11", "mouth_white_12", "mouth_white_13", "mouth_white_14",
                "mouth_white_15", "mouth_white_16", "mouth_white_17", ]
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
    animations_start = dict()
    animations_step = dict()
    animations_status = dict()
    anim_speed_counter = dict()
    animations_showing = dict()

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
        self.new_shape = self.overlays[0]["background"].shape[1], self.overlays[0]["background"].shape[0]

    def get_img(self, part):
        for anim_id, showing in self.animations_showing.items():
            if showing and part == self.animations[anim_id]["replaces"]:
                return self.animations[anim_id]["frames"][self.animations_step[anim_id]]
        if part in self.overlays[self.overlay_id]:
            return self.overlays[self.overlay_id][part]
        else:
            return self.overlays[0][part]

    def shape(self):
        return self.overlays[0]["background"].shape

    def w_s(self):
        return self.new_shape

    def load_image(self, path):
        if path != "":
            return utils.load_image(path)
        else:
            return cp.zeros(self.overlays[0]["background"].shape, dtype=cp.uint8)

    def toggle_animation(self, anim_id):
        overlay_str = '|' + str(self.overlay_id) + '|'
        if overlay_str in self.animations[anim_id]["for_overlays"]:
            if anim_id not in self.animations_status:
                self.animations_status[anim_id] = 1
            if self.animations_status[anim_id] == -1:
                self.animations_step[anim_id] = len(self.animations[anim_id]["frames"]) - 1
            else:
                self.animations_step[anim_id] = 0
            self.anim_speed_counter[anim_id] = -1
            self.animations_start[anim_id] = True
            self.animations_showing[anim_id] = True

    def change_overlay(self, over_id):
        self.overlay_id = over_id
        self.animations_start = dict()
        self.animations_step = dict()
        self.animations_status = dict()
        self.anim_speed_counter = dict()
        self.animations_showing = dict()

    def update_animation(self):
        for anim_id, running in self.animations_start.items():
            if running:
                self.anim_speed_counter[anim_id] += 1
                if self.anim_speed_counter[anim_id] == self.animations[anim_id]["speed"]:
                    self.anim_speed_counter[anim_id] = 0
                    if self.animations_status[anim_id] == 1:
                        if self.animations_step[anim_id] < len(self.animations[anim_id]["frames"]) - 1:
                            self.animations_step[anim_id] += 1
                        else:
                            self.animations_start[anim_id] = False
                            if self.animations[anim_id]["on_end"] != "leave":
                                self.animations_showing[anim_id] = False  # прекратить показ анимации
                            if self.animations[anim_id]["on_repeat"] == "reverse":
                                self.animations_status[anim_id] = -1  # проигрывать в обратном порядке при запуске
                    elif self.animations_status[anim_id] == -1:
                        if self.animations_step[anim_id] > 0:
                            self.animations_step[anim_id] -= 1
                        else:
                            self.animations_start[anim_id] = False
                            self.animations_showing[anim_id] = False  # прекратить показ анимации
                            if self.animations[anim_id]["on_repeat"] == "reverse":
                                self.animations_status[anim_id] = 1  # проигрывать в нормальном порядке при запуске

    def get_cur_animations(self):
        st = ""
        for anim_id, running in self.animations_start.items():
            if running:
                st += str(anim_id) + " "
        if st == "":
            st = "-1"
        return st
