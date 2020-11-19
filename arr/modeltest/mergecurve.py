from tools.getmask import get_mask
import numpy as np
import os
import PIL
import cv2
import re
import sympy as sp

from fitcurve.fitCurves import *


def crosspoint(line1, line2):
    [[vx1], [vy1], [x1], [y1]] = line1
    [[vx2], [vy2], [x2], [y2]] = line2




    A = np.array([[vy1, -vx1], [vy2, -vx2]])
    b = np.array([x1*vy1 - y1*vx1, x2*vy2 - y2*vx2])

    A_inv = np.linalg.inv(A)


    x = np.dot(A_inv, b)


    return x


def chuidian(line, point):
    [[vx], [vy], [x1], [y1]] = line
    [x2, y2] = point

    A = np.array([[vx, vy], [vy, -vx]])
    b = np.array([y2 * vy + x2 * vx, x1 * vy - y1 * vx])

    A_inv = np.linalg.inv(A)

    x = np.dot(A_inv, b)

    #x = x.astype(int)

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



    if length < 50:
        return False


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


def get_bezier(points, error):
    rate = 5
    xrange = range(0, len(points), rate)

    points = points[xrange, :]

    beziers = fitCurve(points, error)
    print('number of beziers:{}'.format(len(beziers)))

    return beziers


def get_linepoint(startpoint, endpoint):


    return


def get_bezier_points(beziers):

    points = np.empty([0, 2])

    for i in range(len(beziers)):
        for t in np.arange(0, 1.01, 0.05):
            bezier_points = tri_bezier(beziers[i], t)
            points = np.append(points, bezier_points, axis=0)

    return points


def get_bezier_line_cross(bezier, line):
    print(line)

    [p1, p2, p3, p4] = bezier
    [[vx], [vy], [x], [y]] = line

    t = sp.Symbol('t')

    parm_1 = (1 - t) ** 3
    parm_2 = 3 * (1 - t) ** 2 * t
    parm_3 = 3 * t ** 2 * (1 - t)
    parm_4 = t ** 3

    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3 + p4[0] * parm_4
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3 + p4[1] * parm_4

    f = py*vx - px*vy + vy*x - vx*y

    t = sp.solve(f)

    print('solutions: {}'.format(t))

    t = t[0]


    #t = 1

    parm_1 = (1 - t) ** 3
    parm_2 = 3 * (1 - t) ** 2 * t
    parm_3 = 3 * t ** 2 * (1 - t)
    parm_4 = t ** 3

    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3 + p4[0] * parm_4
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3 + p4[1] * parm_4


    print(t)

    cp = np.array([px, py]).astype(int)


    return t, cp




def get_perspective(m, cnt):


    ones = np.ones(len(cnt))

    cnt = np.c_[cnt, ones.T]
    cnt = np.dot(m , cnt.T).T[:,0:2]
    #cnt = np.expand_dims(cnt, axis = 1)
    cnt = cnt.astype(int)
    return cnt





base_dir = r'C:\Users\fscut\Desktop\arrtest\data'

data_types = ['testbd', 'test', 'trainbd', 'testraw']

data_dir = os.path.join(base_dir, data_types[-1])

mode = 'single'



filename = 'a37.JPG'

out_dir = os.path.join(data_dir + '_result', filename[0:-4])
os.makedirs(out_dir, exist_ok = True)

file_dir = os.path.join(data_dir, filename)

img = PIL.Image.open(file_dir).convert('RGB')

maskmodes = ['read', 'write']
maskmode = maskmodes[0]

if maskmode == 'read':

    mask = cv2.imread(os.path.join(out_dir, 'mask.png'), cv2.IMREAD_GRAYSCALE)
    if mode != 'single':
        edges = cv2.imread(os.path.join(out_dir,'edges.png'), cv2.IMREAD_GRAYSCALE)

else:


    #mode 1:seg 0:edge
    mask = get_mask(img, rate=4, mode=1)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(out_dir,'mask.png'), mask)

    if mode != 'single':
        edges = cv2.imread(os.path.join(out_dir,'edges.png'), cv2.IMREAD_GRAYSCALE)

    # edges = get_mask(img, rate=4, mode=0)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)
    # cv2.imwrite(os.path.join(out_dir,'edges.png'), edges)



# edges[edges == 255] = 1
# skeleton0 = morphology.skeletonize(edges)
# skeleton = skeleton0.astype(np.uint8) * 255

# cv2.imshow('skeleton', skeleton)
# cv2.waitKey(0)
# cv2.imwrite(os.path.join(out_dir, 'skeleton.png'), skeleton)

if mode != 'single':
    merged = mask.copy()
    merged[edges == 255] = 0
    cv2.imshow('merged', merged)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(out_dir, 'merged.png'), merged)
else:
    merged = mask

#merged = cv2.cvtColor(merged,cv2.COLOR_RGB2GRAY)
cnts0 = cv2.findContours(merged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
#cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测

cnts = cnts0[1]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序

img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
shape = img.shape
img = cv2.resize(img, (int(shape[1]/4), int(shape[0]/4)))



excnt = cnts[0]
cv2.drawContours(img, [excnt], -1 , (0, 0, 255), 2)
cv2.imshow('contour', img)
cv2.waitKey(0)
cv2.imwrite(os.path.join(out_dir, 'contour.png'), img)

txt_dir = os.path.join(data_dir, filename[0:-4] + '.txt')
with open(txt_dir, 'r') as f:
    perrate = f.readline()
    lines = f.read()

numbers = re.split(r',|;|\n',lines)
numbers[0] = numbers[0][1:]
numbers[-1] = numbers[0][0:-1]
numbers.pop(3)
numbers.pop(6)
m = np.array(numbers, dtype = 'float64')
m = np.reshape(m, (3,3))



excnt = np.squeeze(excnt)
per_excnt = get_perspective(m, excnt)
num_points = len(excnt)


min = (np.min(per_excnt[:,0]), np.min(per_excnt[:,1]))
per_excnt[:,0] = per_excnt[:,0] + np.ones(num_points)*(-min[0] + 50)
per_excnt[:,1] = per_excnt[:,1] + np.ones(num_points)*(-min[1] + 50)
print(min)
print(img.shape)

max = (np.max(per_excnt[:,0]), np.max(per_excnt[:,1]))
print(max)

per_excnt = np.expand_dims(per_excnt, axis=1)

edges_new = np.zeros((max[1] + 50, max[0] + 50, 3))
#edges_new = cv2.warpPerspective(img, m, (800,1000))


cv2.drawContours(edges_new, [per_excnt], 0 , (255, 255, 255), 2)
cv2.imshow('contourbd', edges_new)
cv2.waitKey(0)
cv2.imwrite(os.path.join(out_dir, 'contourbd.png'), edges_new)






contours = []
contours.append(per_excnt)

for cnt in cnts[1:]:
    if cv2.arcLength(cnt, False) < 100.0:
        continue

    num_points = len(cnt)
    cnt = np.squeeze(cnt)
    cnt = get_perspective(m, cnt)
    #cnt = np.expand_dims(cnt, axis=1)
    cnt[:, 0] = cnt[:, 0] + np.ones(num_points) * (-min[0] + 50)
    cnt[:, 1] = cnt[:, 1] + np.ones(num_points) * (-min[1] + 50)
    cnt = np.expand_dims(cnt, axis=1)
    point = cnt[0][0]
    if cv2.pointPolygonTest(per_excnt,tuple(point),True) > 0 :
        contours.append(cnt)

    cv2.drawContours(edges_new, [cnt], 0, (255, 255, 255), 2)


cv2.imshow('contourbd', edges_new)
cv2.waitKey(0)
cv2.imwrite(os.path.join(out_dir, 'contourbd.png'), edges_new)




chuixians = []
lineframe = edges_new.copy()

for c in contours:


    peri = cv2.arcLength(c, True)  # 计算轮廓周长
    approx = cv2.approxPolyDP(c, 0.004*peri, True)           # 轮廓多边形拟合

    for peak in approx:

        peak = peak[0]
        print(peak)
        cv2.circle(lineframe, tuple(peak), 10, (255, 0, 0))


    # cv2.imshow('4point', img)
    # cv2.waitKey(0)
    #

    line_curves = []
    ifline = []

    loc0 = 0
    while not np.all((c[loc0] == approx[-1])):
        loc0 = (loc0 + 1) % len(c)

    loc1 = loc0



    startloc = loc1

    m = 0
    for i in range(len(approx)):

        loc0 = loc1

        while not np.all((c[loc1] == approx[i])):
            # print(c[loc0])
            # print(hull[i])

            loc1 = (loc1 + 1) % len(c)


        if loc0 > loc1:
            linepoints = np.vstack((c[loc0:len(c)], c[0: loc1 + 1]))
        else:
            linepoints = c[loc0:loc1 + 1]

        #inepoints = c[loc0:loc1 + 1]

        print('loc0: {}, loc1 {}'.format(loc0, loc1))

        # [vx, vy, x, y] = cv2.fitLine(linepoints, cv2.DIST_L2, 0, 0.01, 0.01)




        line = cv2.fitLine(linepoints, cv2.DIST_L2, 0, 0.01, 0.01)





        #cd1 = chuidian(line, c[loc0])

        #cd2 = chuidian(line, c[loc1])



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

            if m == 1:
                line_curves.append(curve)
                ifline.append(False)
                m = 0


            line_curves.append(line)
            ifline.append(True)
            #cv2.line(lineframe, tuple(cd1), tuple(cd2), (0, 255, 0), 2)


    cv2.imshow("test", lineframe)
    cv2.waitKey(0)




    if m == 1:

        #and np.all(line_curves[0][0] == curve[-1]) and np.all(line_curves[0][1] != curve[0])
        if len(line_curves) > 0 and not ifline[0] :
            print('connecting end curve')
            end_curve = line_curves.pop(0)
            ifline.pop(0)
            curve = np.vstack((curve, end_curve))

        line_curves.append(curve)
        ifline.append(False)


    if ifline[0]:

        if ifline[-1]:

            startpoint = crosspoint(line_curves[0], line_curves[-1])

            lastend = startpoint


        else:
            #beziers = get_bezier(np.squeeze(line_curves[-1]), 10)

            startpoint = chuidian(line_curves[0], line_curves[-1][-1][0])
            chuixians.append((line_curves[-1][-1][0], startpoint))


    else:

        lastend = line_curves[0][0][0]


        #beziers = get_bezier(np.squeeze(line_curves[0]), 10)
        #cdpoint = chuidian(line_curves[-1], beziers[0][0])
        #chuixians.append((cdpoint, beziers[-1][-1]))

        startpoint = line_curves[0][-1]




    for i in range(len(line_curves) - 1):


        if ifline[i]:

            if ifline[i + 1]:

                endpoint = crosspoint(line_curves[i], line_curves[i + 1])

                #linepoints = get_linepoints(startpoint, endpoint)


            else :

                cdpoint = line_curves[i + 1][0][0]

                endpoint = chuidian(line_curves[i], cdpoint)

                chuixians.append((endpoint, cdpoint))



            print(endpoint)
            print(startpoint)

            startpoint = startpoint.astype(int)
            endpoint = endpoint.astype(int)

            cv2.line(lineframe, tuple(startpoint), tuple(endpoint), (0, 255, 0), 2)


            print('line start point：{} endpoint {}'.format(startpoint, endpoint))

            startpoint = endpoint



        else :

            beziers = get_bezier(np.squeeze(line_curves[i]), 10)


            b_points = get_bezier_points(beziers)

            b_points = b_points.astype(int)

            cv2.drawContours(lineframe, [b_points], -1, (0, 0, 255), 2)

            cdpoint = beziers[-1][-1]

            startpoint = chuidian(line_curves[i + 1], cdpoint)

            chuixians.append((cdpoint, startpoint))







    if ifline[-1]:
        cv2.line(lineframe, tuple(startpoint), tuple(lastend), (0, 255, 0), 2)

    else:

        beziers = get_bezier(np.squeeze(line_curves[-1]), 10)

        b_points = get_bezier_points(beziers)

        b_points = b_points.astype(int)

        cv2.drawContours(lineframe, [b_points], -1, (0, 0, 255), 2)




    for chuixian in chuixians:

        cd1 = chuixian[0].astype(int)

        cd2 = chuixian[1].astype(int)

        # chuixian[0][1] = int(chuixian[0][1])
        #
        # chuixian[1][0] = int(chuixian[1][0])
        #
        # chuixian[1][1] = int(chuixian[1][1])

        print(chuixian)

        cv2.line(lineframe, tuple(cd1), tuple(cd2), (255, 0, 0), 2)



    cv2.imshow("test", lineframe)
    cv2.waitKey(0)









        #cv2.circle(lineframe, tuple(c[loc0][0]), 10, (255, 0, 0))
        #cv2.circle(lineframe, tuple(c[loc1][0]), 10, (255, 0, 0))








    # for curve in line_curves:
    #     print('print curve')
    #     # if len(curve ) < 5:
    #     #     continue
    #     #print(curve)
    #     curve = np.squeeze(curve)
    #     beziers, b_points = get_bezier_points(curve, 10)
    #
    #     print(b_points)
    #     cv2.drawContours(lineframe, [b_points], -1 , (0, 0, 255), 2)
    #     cv2.drawContours(save_img, [b_points], -1 , (0, 0, 255), 2)




cv2.imwrite(os.path.join(out_dir, 'curves.jpg'), lineframe)
