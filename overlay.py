import numpy as np

import utils


def get_part_name_for_loading(part):
    if part == "mouth":
        return ["mouth_1", "mouth_2", "mouth_3", "mouth_4", "mouth_5", "mouth_6", "mouth_7", "mouth_8", "mouth_9",
                "mouth_10", "mouth_11", "mouth_12", "mouth_13"]
    elif part == "r_eye_white":
        return ["r_eye_white_0", "r_eye_white_1", "r_eye_white_2", "r_eye_white_3"]
    elif part == "l_eye_white":
        return ["l_eye_white_0", "l_eye_white_1", "l_eye_white_2", "l_eye_white_3"]
    elif part == "r_eye":
        return ["r_eye_0", "r_eye_1", "r_eye_2", "r_eye_3"]
    elif part == "l_eye":
        return ["l_eye_0", "l_eye_1", "l_eye_2", "l_eye_3"]
    else:
        return [part]


class Overlay:
    overlay_id = 0

    def __init__(self, json_overlays):
        self.overlays = []
        for i in range(len(json_overlays)):
            replaces = json_overlays[i]["replaces"]
            self.overlays.append(dict())
            for j in replaces:
                names = get_part_name_for_loading(j)
                for part in names:
                    self.overlays[i][part] = self.load_image(replaces[j])

    def get_img(self, part):
        if part in self.overlays[self.overlay_id]:
            return self.overlays[self.overlay_id][part]
        else:
            return self.overlays[0][part]

    def shape(self):
        return self.overlays[0]["background"].shape

    def w_shape(self):
        new_shape = (self.overlays[0]["background"].shape[1], self.overlays[0]["background"].shape[0])
        return new_shape

    def load_image(self, path):
        if path != "":
            return utils.load_image(path)
        else:
            return np.zeros(self.overlays[0]["background"].shape, dtype=np.uint8)
