#!/usr/bin/env python
import numpy as np
import cv2
import os

path = '//home//rilab-02//workspace//turtlebot3//sign_detector//'
FLANN_INDEX_LSH = 6
orb = cv2.ORB_create()
cap = cv2.VideoCapture(0)
names = ["1avoid", "avoid_t1", "avoid_t2", "avoid_t3", "avoid_t4", "avoid_t5", "avoid_t6", "avoid_t7", "avoid_t8", "avoid_t9", "avoid_t10", "avoid_t11", "avoid_t12", "avoid_t13", "avoid_t14", "avoid_t15", "avoid_t16", "avoid_t17", "avoid_t18", "avoid_t19", "avoid_t20", "avoid_t21", "avoid_t22", "avoid_t23", "avoid_t24", "avoid_t25", "avoid_t26", "avoid_t27", "avoid_t28", "avoid_t29", "avoid_t30",
         "avoid_t31", "avoid_t32", "avoid_t33", "avoid_t34", "avoid_t35", "avoid_t36", "avoid_t37", "avoid_t38","1avoid_t1", "1avoid_t2", "1avoid_t3", "1avoid_t4", "1avoid_t5", "1avoid_t6", "1avoid_t7", "1avoid_t8", "1avoid_t9", "1avoid_t10"]


def draw_match(matrix, matchesMask, qimg, kp_q, kp_t, good_points, name):
    t_img = None
    res = None
    for f_name in names:
        if name is f_name:
            fname = os.path.join(path, f"{name}.png")
            t_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            h, w = t_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            # t_img = cv2.polylines(t_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    res = cv2.drawMatches(qimg, kp_q, t_img, kp_t, good_points, res, **draw_params)
    return res


def matchImage(kp_t, des_t, kp_q, des_q, qimg, name):
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_q, des_t, k=2)
    good_points = []

    for mat in matches:
        if len(mat) > 1:
            m, n = mat
            if m.distance < 0.6*n.distance:
                good_points.append(m)
    if len(good_points) > 10:
        query_pts = np.array([kp_q[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.array([kp_t[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        draw_img = draw_match(matrix, matchesMask, qimg, kp_q, kp_t, good_points, name)
        cv2.imshow("test", draw_img)
        if np.array(matchesMask).sum() > 10:
            return True
        else:
            return False


def detect_object(names, keypoints_train, descriptors_train):
    while True:
        ret, frame = cap.read()
        if cap.isOpened():
            qimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_q, des_q = orb.detectAndCompute(qimg, None)
            for name, kp_t, des_t in zip(names, keypoints_train, descriptors_train):
                result = matchImage(kp_t, des_t, kp_q, des_q, qimg, name)
                cv2.imshow("frame", frame)
                cv2.waitKey(10)
                if result is True:
                    return name

        else:
            print("cameara disable")


def main():
    keypoints_train = []
    descriptors_train = []
    for name in names:
        filename = os.path.join(path, f"{name}.png")
        train_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        kp_t, des_t = orb.detectAndCompute(train_img, None)
        keypoints_train.append(kp_t)
        descriptors_train.append(des_t)

    while True:
        object = detect_object(names, keypoints_train, descriptors_train)
        print(f"{object} sign detected!")
        k = cv2.waitKey(10)
        if k is 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
