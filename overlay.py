import numpy as np

import utils


class Overlay:
    overlay_id = 0

    def __init__(self, json_overlays):
        self.overlays = []
        for i in range(len(json_overlays)):
            replaces = json_overlays[i]["replaces"]
            self.overlays.append(dict())
            for j in replaces:
                self.overlays[i][j] = self.load_image(replaces[j])

    def get_img(self, part):
        if part in self.overlays[self.overlay_id]:
            return self.overlays[self.overlay_id][part]
        else:
            return self.overlays[0][part]

    def shape(self):
        return self.overlays[0]["background"].shape

    def warp_shape(self):
        new_shape = (self.overlays[0]["background"].shape[1], self.overlays[0]["background"].shape[0])
        return new_shape

    def load_image(self, path):
        if path != "":
            return utils.load_image(path)
        else:
            return np.zeros(self.overlays[0]["background"].shape, dtype=np.uint8)
