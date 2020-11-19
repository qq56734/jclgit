import cv2
import numpy as np
import math
import imutils

import os

from qumaoci2 import qumaoci

# img = cv2.imread("C:/AE_PUFF/python_vision/2018_04_27/kk-3.jpg")




base_dir = '../data\stream/1604033805.2423697'



def crosspoint(line1, line2):
    [[vx1], [vy1], [x1], [y1]] = line1
    [[vx2], [vy2], [x2], [y2]] = line2




    A = np.array([[vy1, -vx1], [vy2, -vx2]])
    b = np.array([x1*vy1 - y1*vx1, x2*vy2 - y2*vx2])

    A_inv = np.linalg.inv(A)


    x = np.dot(A_inv, b)


    return x




#skeletonqumaoci
edges = cv2.imread(os.path.join(base_dir, 'skeleton.png'), cv2.IMREAD_GRAYSCALE)

edges = qumaoci(edges)

# cv2.imshow('qumaoci', edges)
#
# cv2.waitKey(0)


#lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
result = cv2.imread(os.path.join(base_dir, 'frame.jpg'))



cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
rows, cols = edges.shape

edge_center = np.array([cols/2, rows/2])











if len(cnts) > 0:
    cnts = sorted(cnts, key= lambda c: cv2.arcLength(c, False), reverse=True ) # 根据轮廓面积从大到小排序

    for cnt in cnts[0:6]:

        cnt_s = np.squeeze(cnt)
        center_p = cnt_s.mean(axis=0)
        print(center_p)



        if np.linalg.norm(center_p - edge_center) < 200:
            #print(cnt)
            c = cnt
            # cv2.drawContours(result,[cnt],0,(0,255,0),5)
            # cv2.imshow('result', result)
            # cv2.waitKey(0)

    peri = cv2.arcLength(c, True)    # 计算轮廓周长



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



for i in range(0, 4):

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

    lines.append(line)


    copy = result.copy()

    [vx, vy, x, y] = line

    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    img = cv2.line(copy, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    cv2.circle(copy, tuple(c[loc0][0]), 10, (255, 0, 0))
    cv2.circle(copy, tuple(c[loc1][0]), 10, (255, 0, 0))

    cv2.drawContours(copy, [linepoints], 0, (0, 0, 255), 5)
    #
    #
    #
    #
    cv2.imshow("test", copy)
    cv2.waitKey(0)

crosspoints = []
for i in range(-1, 3):
    crossp = crosspoint(lines[i], lines[i + 1])
    crosspoints.append(crossp)




for peak in crosspoints:

    #peak = peak[0]
    peak = peak.astype(int)
    print(peak)
    cv2.circle(result, tuple(peak), 6, (0, 0, 255))
    # cv2.imshow('4point', result)
    #
    # cv2.waitKey(0)






cv2.imwrite(os.path.join(base_dir, 'linecrosspoints.jpg'), result)
#cv2.imwrite(os.path.join(base_dir, 'lines.jpg'), detectlines)
#cv2.imwrite(os.path.join(base_dir, 'centers.jpg'), centerresult)