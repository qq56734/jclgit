import cv2
import numpy as np
import math
import os
from PIL import Image, ExifTags


def get_rank(points):
    points = points[np.argsort(points[:, 0])]

    if points[0][1] > points[1][1]:

        tmp = points[0].copy()
        points[0] = points[1].copy()


    else:

        tmp = points[1].copy()

    if points[2][1] > points[3][1]:
        points[1] = points[3].copy()
        points[3] = points[2].copy()
        points[2] = tmp

    else:

        points[1] = points[2].copy()
        points[2] = tmp

    width = np.linalg.norm(points[1] - points[0])
    length = np.linalg.norm(points[0] - points[2])

    return width > length, points







def refine(point, direction, ran, maxextend = 5):

    x = point[0]
    y = point[1]

    # numpoint = 0
    #
    #
    # for i in range(1, ran):
    #     i = direction[1]*i*(-1)
    #     if edges[y+i][x] == 255:
    #         numpoint += 1

    #refined = not numpoint > 0

    refined = False

    extend = 0

    while not refined and extend < maxextend:
        x += direction[0]
        numpoint = 0

        for i in range(1, ran):
            i = direction[1]*i*(-1)
            if edges[y+i][x] == 255:
                numpoint += 1

        refined = not numpoint > 0

        extend += 1

    point[0] = x - direction[0]

    #numpoint = 0
    ran += extend - 1

    extend = 0

    refined = False

    # for i in range(1, ran):
    #
    #     i = direction[0]*i*(-1)
    #     if edges[y][x+i] == 255:
    #         numpoint += 1
    #
    # refined = not numpoint > 0




    while not refined and extend < maxextend:
        y += direction[1]
        numpoint = 0
        for i in range(1, ran):
            i = direction[0]*i*(-1)
            if edges[y][x+i] == 255:
                numpoint += 1
        refined = not numpoint > 0

        extend += 1



    point[1] = y - direction[1]



    return point




















base_dir = r'data\stream\1603950333.7929368'

#skeletonqumaoci
#edges = cv2.imread(os.path.join(base_dir, 'skeleton.png'), cv2.IMREAD_GRAYSCALE)

edges = Image.open(os.path.join(base_dir, 'skeletonqumaoci.png')).convert('L')


edges = np.array(edges)

img = Image.open(os.path.join(base_dir, 'frame.jpg'))

img = np.array(img)


cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
cnts = cnts[1]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序


c = cnts[0]
peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
approx = cv2.approxPolyDP(c, 0.02*peri, True)


# edges =cv2.drawContours(edges,cnts,-1,(255,0,0),1)  # img为三通道才能显示轮廓
#
# for point in cnts[0]:
#     cv2.circle(edges, tuple(point[0]), 6, (255, 0, 0))
points = np.zeros((4, 2), dtype=int)

i = 0

for peak in approx:
    peak = peak[0]
    points[i] = peak
    #cv2.circle(edges, tuple(peak), 6, (255, 0, 0))
    i += 1

w, points = get_rank(points)

print(points)


direction = {0:(-1, -1), 1:(1, -1), 2:(-1,1), 3:(1, 1)}
ran = 4


for i in range(4):


    points[i] = refine(points[i], direction[i], ran)

    cv2.circle(img, tuple(points[i]), 6, (0, 0, 255))

    print(points[i])


cv2.imwrite(os.path.join(base_dir, 'refined.jpg'), img)

cv2.imshow('drawimg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()









