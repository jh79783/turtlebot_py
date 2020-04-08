import copy
import glob
import cv2
import numpy as np
import time
train_files = dict()

train_files["avoid"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-detect_sign//data/avoid*.png")
train_files["park"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-detect_sign//data/park*.png")
train_files["uturn"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-detect_sign//data/uturn*.png")
train_files["dont"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-detect_sign//data/dont*.png")
train_files["tun"] = glob.glob("C://Users//Mun//Downloads//turtlebot3-detect_sign//data/tun*.png")

selected_files_point = dict()

FLANN_INDEX_LSH = 6

qimg = {"avoid": ["C://Users//Mun//Downloads//turtlebot3-detect_sign//data/avoid_o8.png",
                  "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/avoid_t23.png",
                  "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/avoid_t27.png"],
        "park": ["C://Users//Mun//Downloads//turtlebot3-detect_sign//data/park_o2.png",
                 "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/park_o1.png",
                 "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/park_o7.png"],
        "uturn": ["C://Users//Mun//Downloads//turtlebot3-detect_sign//data/uturn_t17.png",
                  "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/uturn_t38.png",
                  "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/uturn_t13.png"],
        "dont": ["C://Users//Mun//Downloads//turtlebot3-detect_sign//data/dont_t37.png",
                 "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/dont_t29.png",
                 "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/dont_t13.png"],
        "tun": ["C://Users//Mun//Downloads//turtlebot3-detect_sign//data/tun_t33.png",
                "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/tun_t3.png",
                "C://Users//Mun//Downloads//turtlebot3-detect_sign//data/tun_t12.png"]}

def main():
    same_category_files = None

    for category, image_files in train_files.items():
        other_category_files = []
        point = []
        start = time.time()
        for other_categ, other_files in train_files.items():
            if other_categ == category:
                continue
            other_category_files += other_files

        for qimg_file in image_files:
            same_category_files = copy.deepcopy(image_files)
            same_category_files.remove(qimg_file)
        for qfiles in list(qimg[category]):
            # print("same")

            TP, true_matched_files = match_images(qfiles, same_category_files)
            # print("other")
            FP, false_matched_files = match_images(qfiles, other_category_files)
            point.append((TP, FP))
        end = time.time()
        elapsed = end - start
        print(category, "time: ", elapsed)
        selected_files_point[category] = point
        print(selected_files_point)

def match_images(qimg, img_file):
    T_count = 0
    F_count = 0
    file = None
    # print("q:", qimg)
    orb = cv2.ORB_create()
    q_img = cv2.imread(qimg, cv2.COLOR_BGR2GRAY)

    kp_q, des_q = orb.detectAndCompute(q_img, None)

    for file in img_file:
        img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        kp_i, des_i = orb.detectAndCompute(img, None)
        # print(file)
        TF = fla(kp_q, kp_i, des_i, des_q)
        cv2.imshow("qimg", q_img)
        cv2.imshow("img", img)
        cv2.waitKey(10)
        if TF is True:
            T_count += 1
            file = qimg
        else:
            F_count += 1
            file = qimg
    entitle_file = len(img_file)
    TP = TPC(entitle_file, T_count)
    return TP, file

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




if __name__ == "__main__":
    main()

