import cupy as cp

import utils


def get_part_name_for_loading(part):
    if part == "mouth":
        return ["mouth_0", "mouth_1", "mouth_2", "mouth_3", "mouth_4", "mouth_5", "mouth_6", "mouth_7", "mouth_8",
                "mouth_9", "mouth_10", "mouth_11", "mouth_12", "mouth_13", "mouth_14", "mouth_15", "mouth_16",
                "mouth_17"]
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

    def __init__(self, json_overlays, json_animations, json_limits):
        self.overlays = []
        for i in range(len(json_overlays)):
            replaces = json_overlays[i]["replaces"]
            self.overlays.append(dict())
            for j in replaces:
                names = get_part_name_for_loading(j)
                for part in names:
                    self.overlays[i][part] = self.load_image(replaces[j])
        self.anim = []
        for i in range(len(json_animations)):
            self.anim.append(dict())
            for j in json_animations[i]:
                self.anim[i][j] = json_animations[i][j]
            frames = json_animations[i]["frames"]
            self.anim[i]["frames"] = []
            for j in range(len(frames)):
                self.anim[i]["frames"].append(self.load_image(frames[j]))
        self.new_shape = self.overlays[0]["background"].shape[1], self.overlays[0]["background"].shape[0]
        self.limits = dict()
        self.rots = dict()
        for i in range(len(json_limits)):
            part = json_limits[i]["part"]
            if "limits" in json_limits[i]:
                limits = json_limits[i]["limits"]
                self.limits[part] = dict()
                for j in limits:
                    self.limits[part][j] = limits[j]
            if "rot_point" in json_limits[i]:
                points = json_limits[i]["rot_point"]
                self.rots[part] = dict()
                for j in points:
                    self.rots[part][j] = points[j]
        self.bbs = []
        for i in range(len(json_overlays)):
            replaces = json_overlays[i]["replaces"]
            self.bbs.append(dict())
            for j in replaces:
                names = get_part_name_for_loading(j)
                for part in names:
                    self.bbs[i][part] = utils.get_bb(self.overlays[i][part])
                    self.bbs[i][part] = update_bb(
                        self.bbs[i][part],
                        (self.lim(part, "y_min"), self.lim(part, "y_max")),
                        (self.lim(part, "x_min"), self.lim(part, "x_max")),
                        self.rp(part),
                        (self.lim(part, "rot_min"), self.lim(part, "rot_max")),
                        self.shape()
                    )
        for i in range(len(json_animations)):
            self.anim[i]["bb"] = utils.get_bb(self.anim[i]["frames"][0])
            part = self.anim[i]["replaces"]
            self.anim[i]["bb"] = update_bb(
                self.anim[i]["bb"],
                (self.lim(part, "y_min"), self.lim(part, "y_max")),
                (self.lim(part, "x_min"), self.lim(part, "x_max")),
                self.rp(part),
                (self.lim(part, "rot_min"), self.lim(part, "rot_max")),
                self.shape()
            )

    def get_bb(self, part):
        for anim_id, showing in self.animations_showing.items():
            if showing and part == self.anim[anim_id]["replaces"]:
                return self.anim[anim_id]["bb"]
        if part in self.overlays[self.overlay_id]:
            return self.bbs[self.overlay_id][part]
        else:
            return self.bbs[0][part]

    def rp(self, part):
        if part in self.rots:
            return [self.rots[part]["x"], self.rots[part]["y"]]
        return [0, 0]

    def rp_with_bb(self, part):
        if part in self.rots:
            return [self.rots[part]["x"] - self.get_bb(part)[2], self.rots[part]["y"] - self.get_bb(part)[0]]
        return [0, 0]

    def lim(self, part, limit):
        if part in self.limits:
            if limit in self.limits[part]:
                return self.limits[part][limit]
        return 0

    def get_img(self, part):
        for anim_id, showing in self.animations_showing.items():
            if showing and part == self.anim[anim_id]["replaces"]:
                return self.anim[anim_id]["frames"][self.animations_step[anim_id]]
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
        if overlay_str in self.anim[anim_id]["for_overlays"]:
            if anim_id not in self.animations_status:
                self.animations_status[anim_id] = 1
            if self.animations_status[anim_id] == -1:
                self.animations_step[anim_id] = len(self.anim[anim_id]["frames"]) - 1
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
                if self.anim_speed_counter[anim_id] == self.anim[anim_id]["speed"]:
                    self.anim_speed_counter[anim_id] = 0
                    if self.animations_status[anim_id] == 1:
                        if self.animations_step[anim_id] < len(self.anim[anim_id]["frames"]) - 1:
                            self.animations_step[anim_id] += 1
                        else:
                            self.animations_start[anim_id] = False
                            if self.anim[anim_id]["on_end"] != "leave":
                                self.animations_showing[anim_id] = False  # прекратить показ анимации
                            if self.anim[anim_id]["on_repeat"] == "reverse":
                                self.animations_status[anim_id] = -1  # проигрывать в обратном порядке при запуске
                    elif self.animations_status[anim_id] == -1:
                        if self.animations_step[anim_id] > 0:
                            self.animations_step[anim_id] -= 1
                        else:
                            self.animations_start[anim_id] = False
                            self.animations_showing[anim_id] = False  # прекратить показ анимации
                            if self.anim[anim_id]["on_repeat"] == "reverse":
                                self.animations_status[anim_id] = 1  # проигрывать в нормальном порядке при запуске

    def get_cur_animations(self):
        st = ""
        for anim_id, running in self.animations_start.items():
            if running:
                st += str(anim_id) + " "
        if st == "":
            st = "-1"
        return st


def update_bb(cur, y, x, rot, rot_lim, shape):
    if rot_lim != (0, 0):
        a1 = utils.length((cur[0], cur[3]), rot)
        a2 = utils.length((cur[0], cur[2]), rot)
        a3 = utils.length((cur[1], cur[3]), rot)
        a4 = utils.length((cur[1], cur[2]), rot)
        a = max(a1, a2, a3, a4)
        w_r = rot[0] + a - cur[1]
        w_l = cur[0] - rot[0] + a
        h_u = rot[1] + a - cur[3]
        h_d = cur[2] - rot[1] + a
    else:
        w_r = 0
        w_l = 0
        h_u = 0
        h_d = 0
    cur[0] += 3 * y[0] - int(w_l)
    cur[1] += 3 * y[1] + int(w_r)
    cur[2] += 3 * x[0] - int(h_d)
    cur[3] += 3 * x[1] + int(h_u)
    if cur[0] < 0:
        cur[0] = 0
    if cur[1] > shape[0]:
        cur[1] = shape[0]
    if cur[2] < 0:
        cur[2] = 0
    if cur[3] > shape[1]:
        cur[3] = shape[1]
    return cur
