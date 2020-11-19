import os
import cv2
import numpy as np
import imutils
import math
#import bezier


from fitcurve.fitCurves import *




def chuidian(line, point):
    [[vx], [vy], [x1], [y1]] = line
    [x2, y2] = point[0]

    A = np.array([[vx, vy], [vy, -vx]])
    b = np.array([y2 * vy + x2 * vx, x1 * vy - y1 * vx])

    A_inv = np.linalg.inv(A)

    x = np.dot(A_inv, b)

    x = x.astype(int)

    return x

def get_error(line, linepoints):

    error = 0

    [[vx], [vy], [x1], [y1]] = line

    A = np.array([[vx, vy], [vy, -vx]])

    A_inv = np.linalg.inv(A)

    for point in linepoints:
        [x2, y2] = point[0]
        b = np.array([y2 * vy + x2 * vx, x1 * vy - y1 * vx])
        cd = np.dot(A_inv, b)
        error += np.linalg.norm(cd - point[0])

    error  = error/len(linepoints)
    return error



def is_curve(linepoints, line):



    length = cv2.arcLength(linepoints, False)
    # dis = np.linalg.norm(start - end)
    print('length: {}'.format(length))

    error = get_error(line, linepoints)

    print('error:{}'.format(error) )

    print('error/length:{}'.format(error/length))




    #return error > 0.25  and length < 50

    #error > 0.7

    if length < 20:
        return True


    return error/length > 0.01


def tri_bezier(bezier, t):
    [p1, p2, p3, p4] = bezier
    parm_1 = (1 - t) ** 3
    parm_2 = 3 * (1 - t) ** 2 * t
    parm_3 = 3 * t ** 2 * (1 - t)
    parm_4 = t ** 3

    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3 + p4[0] * parm_4
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3 + p4[1] * parm_4

    return np.array([[px, py]])


def get_bezier_points(points, error):
    rate = 5
    xrange = range(0, len(points), rate)

    points = points[xrange, :]

    beziers = fitCurve(points, error)
    print('number of beziers:{}'.format(len(beziers)))


    points = np.empty([0,2])

    for bezier in beziers:
        for t in np.arange(0, 1.01, 0.05):
            bezier_point = tri_bezier(bezier, t)
            points = np.append(points, bezier_point, axis = 0)

    points = points.astype(int)
    return points





data_dir = r'C:\Users\fscut\Desktop\newresults\deeplabdrn\deeplab + drn epoch 10test'


filenames = os.listdir(os.path.join(data_dir, 'validation'))



filename = 'e25.jpg'


file_dir = os.path.join(data_dir, 'validation',filename)

out_dir = r'C:\Users\fscut\Desktop\linedetect'
out_dir = os.path.join(out_dir, filename[0:-4])
os.makedirs(out_dir, exist_ok = True)


img = cv2.imread(file_dir)
edges = cv2.imread(os.path.join(data_dir, 'prediction',  filename[0:-4] + '.png'), 0)

# cv2.imshow('edge',edges)
# cv2.waitKey(0)


resizeshape = edges.shape


img = cv2.resize(img, (resizeshape[1], resizeshape[0]))





cnts0 = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
#cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

hierarchy = cnts0[-1]
cnts = cnts0[1]
rows, cols = edges.shape

edge_center = np.array([cols/2, rows/2])


cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序







dis = 10000000
contour_n = 0

for i in range(min(len(cnts), 1)):

    cnt_s = np.squeeze(cnts[i])

    center_p = cnt_s.mean(axis=0)

    new_dis = np.linalg.norm(center_p - edge_center)

    if new_dis < dis :
        # print(cnt)
        dis  = new_dis
        excontour = cnts[i]




contours = []
contours.append(excontour)

for cnt in cnts:
    point = cnt[0][0]
    if cv2.pointPolygonTest(excontour,tuple(point),True) > 0 :
        if cv2.arcLength(cnt, False) > 50.0:
            contours.append(cnt)




for c in contours:

    peri = cv2.arcLength(c, True)  # 计算轮廓周长
    approx = cv2.approxPolyDP(c, 0.005*peri, True)           # 轮廓多边形拟合

    for peak in approx:

        peak = peak[0]
        print(peak)
        cv2.circle(img, tuple(peak), 10, (255, 0, 0))

    # cv2.imshow('4point', img)
    # cv2.waitKey(0)
    #

    loc0 = 0
    while not np.all((c[loc0] == approx[-1])):
        loc0 = (loc0 + 1) % len(c)

    loc1 = loc0

    lines = []

    lineframe = img.copy()



    curves = []
    m = 0
    for i in range(len(approx)):

        loc0 = loc1

        while not np.all((c[loc1] == approx[i])):
            # print(c[loc0])
            # print(hull[i])

            loc1 = (loc1 + 1) % len(c)

        # if np.linalg.norm(c[loc1][0] - c[loc0][0]) < 30:
        #
        #
        #     print('loc0: {}, loc1 {}'.format(loc0, loc1))
        #
        #     continue

        # if loc0 < loc1:
        #     linepoints = np.vstack((c[loc1:len(c)], c[0: loc0 + 1]))
        # else:
        #     linepoints = c[loc1:loc0 + 1]

        if loc0 > loc1:
            linepoints = np.vstack((c[loc0:len(c)], c[0: loc1 + 1]))
        else:
            linepoints = c[loc0:loc1 + 1]

        #inepoints = c[loc0:loc1 + 1]

        print('loc0: {}, loc1 {}'.format(loc0, loc1))

        # [vx, vy, x, y] = cv2.fitLine(linepoints, cv2.DIST_L2, 0, 0.01, 0.01)




        line = cv2.fitLine(linepoints, cv2.DIST_L2, 0, 0.01, 0.01)



        lines.append(line)

        cd1 = chuidian(line, c[loc0])

        cd2 = chuidian(line, c[loc1])


        linepoints = linepoints

        if is_curve(linepoints, line):
            print('curve')
            #cv2.line(lineframe, tuple(cd1), tuple(cd2), (0, 0, 255), 2)
            if m == 0:
                curve = linepoints
                m = 1
            else:
                curve = np.vstack((curve, linepoints))
            #print(curve)
        else:
            cv2.line(lineframe, tuple(cd1), tuple(cd2), (0, 255, 0), 2)
            if m == 1:
                curves.append(curve)
                m = 0

    cv2.imshow("test", lineframe)
    cv2.waitKey(0)



    if m == 1:

        if len(curves) > 0 and np.all(curves[0][0] == curve[-1]) and np.all(curves[0][1] != curve[0]):
            end_curve = curves.pop(0)
            curve = np.vstack((curve, end_curve))


        curves.append(curve)
        cv2.circle(lineframe, tuple(c[loc0][0]), 10, (255, 0, 0))
        cv2.circle(lineframe, tuple(c[loc1][0]), 10, (255, 0, 0))





    cv2.imshow("test", lineframe)
    cv2.waitKey(0)



    for curve in curves:
        print('print curve')
        #print(curve)
        curve = np.squeeze(curve)
        b_points = get_bezier_points(curve, 10)

        #print(b_points)
        cv2.drawContours(lineframe, [b_points], -1 , (0, 0, 255), 2)
        #cv2.drawContours(lineframe, [curve], -1, (0, 0, 255), 2)


    cv2.imshow("test", lineframe)
    cv2.waitKey(0)

    cv2.imwrite(os.path.join(out_dir, 'curves.jpg'), lineframe)