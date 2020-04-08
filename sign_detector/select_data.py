import copy
import glob
import cv2
import numpy as np
path = 'C://Users//Mun//Downloads//turtlebot3-rearrange-repository//data'
train_files = dict()


train_files["avoid"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-rearrange-repository//data/avoid*.png")
train_files["park"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-rearrange-repository//data/park*.png")

selected_files = dict()
TARGET_COUNT_PER_CATEGORY = 3
FLANN_INDEX_LSH = 6


def main():
    remained_files = None
    for category, image_files in train_files.items():
        other_category_files = []

        for other_categ, other_files in train_files.items():
            if other_categ == category:
                continue
            other_category_files += other_files

        best_score = 0
        best_file = None
        selected_categ_files = []
        for k in range(TARGET_COUNT_PER_CATEGORY):
            for qimg_file in image_files:
                same_category_files = copy.deepcopy(image_files)
                same_category_files.remove(qimg_file)

                TP, FP, true_matched_files = match_images(qimg_file, same_category_files)
                # print("TP:", TP)
                TP_other, FP_other, false_matched_files = match_images(qimg_file, other_category_files)
                # print("other:", TP_other)
                if best_score < TP - TP_other:
                    best_score = TP - TP_other
                    best_file = qimg_file
            # print("image_files:", image_files)
            selected_categ_files.append(best_file)

            # print("bestfile", best_file)
            TP, FP, true_matched_files = match_images(best_file, image_files)
            best_file = None
            best_score = 0
            # print(true_matched_files)
            if true_matched_files is not None:
                remained_files = [file for file in image_files if file not in true_matched_files]
                image_files = remained_files
            else:
                print("please modify datafile")
            # print("re", remained_files)

        selected_files[category] = selected_categ_files
        print("select_data:", selected_files, "\n")


def match_images(qimg, img_file):
    T_count = 0
    F_count = 0
    file = None
    F_file = None
    orb = cv2.ORB_create()
    q_img = cv2.imread(qimg, cv2.COLOR_BGR2GRAY)
    kp_q, des_q = orb.detectAndCompute(q_img, None)
    for file in img_file:
        img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        kp_i, des_i = orb.detectAndCompute(img, None)
        TF = fla(kp_q, kp_i, des_i, des_q)
        # cv2.imshow("qimg", q_img)
        # cv2.imshow("img", img)
        # cv2.waitKey(10)
        if TF is True:
            T_count += 1
            file = qimg
        else:
            F_count += 1
            file = qimg
    entitle_file = len(img_file)
    TP = TPC(entitle_file, T_count)
    FP = FPC(entitle_file, F_count)
    return TP, FP, file


def fla(kp_q, kp_i, des_i, des_q):
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_q, des_i, k=2)
    good_points = []
    for mat in matches:
        if len(mat) > 1:
            m, n = mat
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
    if len(good_points) > 10:
        query_pts = np.array([kp_q[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.array([kp_i[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if np.array(matchesMask).sum() > 10:
            return True
        else:
            return False
    else:
        return False


def TPC(entitle_file, T_count):
    TP = T_count/entitle_file
    return TP*100


def FPC(entitle_file, F_count):
    FP = F_count/entitle_file
    return FP*100


if __name__ == "__main__":
    main()