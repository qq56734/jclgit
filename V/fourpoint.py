import cv2
import numpy as np
import math
import imutils
from getmask import get_mask
import os
from skimage import morphology

from qumaoci2 import qumaoci

# img = cv2.imread("C:/AE_PUFF/python_vision/2018_04_27/kk-3.jpg")








def crosspoint(line1, line2):
    [[vx1], [vy1], [x1], [y1]] = line1
    [[vx2], [vy2], [x2], [y2]] = line2




    A = np.array([[vy1, -vx1], [vy2, -vx2]])
    b = np.array([x1*vy1 - y1*vx1, x2*vy2 - y2*vx2])

    A_inv = np.linalg.inv(A)


    x = np.dot(A_inv, b)


    return x

def get_4point(frame, out_dir):


    print('getting four point')

    resizeshape = (640, 480)

    mask = get_mask(frame, resizeshape)

    frame = cv2.resize(frame, resizeshape)

    cv2.imwrite(os.path.join(out_dir, 'mask.png'), mask)
    #
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    #
    # cv2.imwrite(os.path.join(out_dir, 'mask_dilate.png'), mask)

    mask[mask == 255] = 1
    skeleton0 = morphology.skeletonize(mask)
    skeleton = skeleton0.astype(np.uint8) * 255


    cv2.imshow('skeleton', skeleton)
    cv2.waitKey(0)


    cv2.imwrite(os.path.join(out_dir, 'skeleton.png'), skeleton)

    edges = qumaoci(skeleton, frame)

    cv2.imshow('skeletonqumaoci', edges)
    cv2.waitKey(0)

    cv2.imwrite(os.path.join(out_dir, 'skeletonqumaoci.png'), edges)




    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    rows, cols = edges.shape

    edge_center = np.array([cols/2, rows/2])

    dis = 10000000

    for i in range(3):

        cnt_s = np.squeeze(cnts[i])

        center_p = cnt_s.mean(axis=0)

        new_dis = np.linalg.norm(center_p - edge_center)

        if new_dis < dis:
            # print(cnt)
            dis = new_dis
            c = cnts[i]



    peri = cv2.arcLength(c, True)  # 计算轮廓周长
    approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合

    #print(cnts)

    hull = cv2.convexHull(approx)

    # for peak in hull:
    #
    #     peak = peak[0]
    #     print(peak)
    #     cv2.circle(result, tuple(peak), 10, (255, 0, 0))
    #     cv2.imshow('4point', result)
    #
    #     cv2.waitKey(0)
    # #


    loc0 = 0
    while not np.all((c[loc0]  == hull[-1])):

            loc0 = (loc0 + 1) % len(c)

    loc1 = loc0

    lines = []

    lineframe = frame.copy()

    for i in range(len(hull)):

        loc0 = loc1

        while not np.all((c[loc1]  == hull[i])):
            #print(c[loc0])
            #print(hull[i])


            loc1 = (loc1 + 1) % len(c)



        # if np.linalg.norm(c[loc1][0] - c[loc0][0]) < 30:
        #
        #
        #     print('loc0: {}, loc1 {}'.format(loc0, loc1))
        #
        #     continue

        if loc0 < loc1:
            linepoints = np.vstack((c[loc1:len(c)], c[0: loc0 + 1]))
        else:
            linepoints = c[loc1:loc0 + 1]

        # if loc0 > loc1:
        #     linepoints = np.vstack((c[loc0:len(c)], c[0: loc1 + 1]))
        # else:
        #     linepoints = c[loc0:loc1 + 1]


        #linepoints = c[loc0:loc1 + 1]


        print('loc0: {}, loc1 {}'.format(loc0, loc1))


        #[vx, vy, x, y] = cv2.fitLine(linepoints, cv2.DIST_L2, 0, 0.01, 0.01)

        line = cv2.fitLine(linepoints, cv2.DIST_L2, 0, 0.01, 0.01)

        [vx, vy, x, y] = line

        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(lineframe, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)

        lines.append(line)


        # copy = result.copy()
        #
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((cols - x) * vy / vx) + y)
        # img = cv2.line(copy, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
        # cv2.circle(copy, tuple(c[loc0][0]), 10, (255, 0, 0))
        # cv2.circle(copy, tuple(c[loc1][0]), 10, (255, 0, 0))
        #
        #cv2.drawContours(lineframe, [linepoints], 0, (0, 0, 255), 2)
        #
        #
        #
        #
        # cv2.imshow("test", copy)
        # cv2.waitKey(0)



    cv2.imwrite(os.path.join(out_dir, 'lines.jpg'), lineframe)

    cv2.imshow('lines', lineframe)

    cv2.waitKey(0)

    crosspoints = []
    for i in range(-1, 3):
        crossp = crosspoint(lines[i], lines[i + 1])
        crosspoints.append(crossp)



    for peak in crosspoints:

        #peak = peak[0]
        peak = peak.astype(int)
        #print(peak)
        cv2.circle(frame, tuple(peak), 6, (0, 0, 255))
        # cv2.imshow('4point', result)
        #
        # cv2.waitKey(0)

    cv2.imshow('4point', frame)

    cv2.waitKey(0)



    cv2.imwrite(os.path.join(out_dir, 'line4point.jpg'), frame)
    #cv2.imwrite(os.path.join(base_dir, 'lines.jpg'), detectlines)
    #cv2.imwrite(os.path.join(base_dir, 'centers.jpg'), centerresult)

    return np.array(crosspoints)