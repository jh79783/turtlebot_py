import glob
import os


class SelectParams:
    def __init__(self, min_pt, inlier_ratio):
        self.min_pt = min_pt
        self.inlier_ratio = inlier_ratio


class Data:
    def __init__(self):
        self.train_files = dict()
        self.selected_files = dict()

        path = "D:/py-project/data"
        self.train_files["avoid"] = glob.glob(os.path.join(path, "avoid_*.png"))
        self.train_files["park"] = glob.glob(os.path.join(path, "park_*.png"))
        self.train_files["uturn"] = glob.glob(os.path.join(path, "uturn_*.png"))
        self.train_files["dont"] = glob.glob(os.path.join(path, "dont_*.png"))
        self.train_files["tun"] = glob.glob(os.path.join(path, "tun_*.png"))

        select_path = "D:/py-project/select_data"
        self.selected_files["avoid"] = glob.glob(os.path.join(select_path, "avoid*.png"))
        self.selected_files["park"] = glob.glob(os.path.join(select_path, "park*.png"))
        self.selected_files["uturn"] = glob.glob(os.path.join(select_path, "uturn*.png"))
        self.selected_files["dont"] = glob.glob(os.path.join(select_path, "dont*.png"))
        self.selected_files["tun"] = glob.glob(os.path.join(select_path, "tun*.png"))

    def tfa(self):
        pass

    def sfa(self):
        pass

def set_params(category):
    param = {"avoid": SelectParams(10, 5.0),
              "park": SelectParams(4, 1000.0),
              "uturn": SelectParams(10, 5.0),
              "dont": SelectParams(10, 5.0),
              "tun": SelectParams(10, 5.0),
             }
    min_pt = param[category].min_pt
    inlier_ratio = param[category].inlier_ratio
    return min_pt, inlier_ratio


