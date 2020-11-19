import cv2
import numpy as np
import math
import os

# img = cv2.imread("C:/AE_PUFF/python_vision/2018_04_27/kk-3.jpg")


def CrossPoint(line1, line2, shape):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3])
    p4 = np.array([x4, y4])



    d1 = np.linalg.norm(p1 - p3)
    d2 = np.linalg.norm(p1 - p4)
    d3 = np.linalg.norm(p2 - p3)
    d4 = np.linalg.norm(p2 - p4)

    mindis = min(min(min(d1, d2), d3), d4)
    print(mindis)
    if mindis > 30:
        return None

    arr_0 = p2 - p1
    arr_1 = p4 - p3

    k1 = arr_0[1]/arr_0[0]*1.0
    k2 = arr_1[1]/arr_1[0]*1.0

    rate = abs(k1/k2)

    if rate < 1.01 and rate > 0.99:
        return None




    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))

    print("cosf{}".format(cos_value))

    angle = np.arccos(cos_value) * (180 / np.pi)

    print(angle)


    if angle < 80 or angle > 100 :
        return None

    print("{},{},{},{},{},{},{},{}".format(x1,y1, x2, y2, x3, y3, x4, y4))

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
        x = x3
        #k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
        y = k1 * x * 1.0 + b1 * 1.0
    elif (x2 - x1) == 0:
        k1 = None
        b1 = 0
        x = x1
        #k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        y = k2 * x * 1.0 + b2 * 1.0
    else:
        #k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        #k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0

    # if np.isinf(x) or np.isinf(y):
    #     return None

    return (int(x), int(y))


base_dir = '../data/webwxgetmsgimg (4)'

#skeletonqumaoci
edges = cv2.imread(os.path.join(base_dir, 'skeletonqumaoci.png'), cv2.IMREAD_GRAYSCALE)

#lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
result = cv2.imread(os.path.join(base_dir, 'raw.jpg'))



minLineLength = 1  # height/32
maxLineGap = 1 # height/40
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength, maxLineGap)


print(len(lines))


crosspoints = []
detectlines = result.copy()
centerresult = result.copy()


for i in range(len(lines) - 1):

    for j in range(i+1, len(lines)):



        crosspoint = CrossPoint(lines[i][0], lines[j][0], edges.shape)

        if crosspoint != None:


            cp = np.array(crosspoint)

            crosspoints.append(cp)



        cv2.circle(result, crosspoint, 6, (255, 0, 0))






    for x1, y1, x2, y2 in lines[i]:
        cv2.line(detectlines, (x1, y1), (x2, y2), (0, 255, 0), 1)
x = np.array(crosspoints)
print(x)



cv2.imshow('result', detectlines)
cv2.waitKey(0)

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)   #n_clusters:number of cluster
kmeans.fit(x)
print(kmeans.labels_)
labels = kmeans.labels_

centers = kmeans.cluster_centers_.astype(int)
print(centers)


color = {1:(255, 0 ,0), 2:(0, 255, 0), 3:(0,0,255), 0:(255,255,255)}

#
# for i in range(len(labels)):
#     cv2.circle(result, tuple(x[i]), 6, color[labels[i]])

for i in range(4):
     cv2.circle(centerresult, tuple(x[i]), 6, color[labels[i]])
     cv2.circle(centerresult, tuple(centers[i]), 6, color[i])



cv2.imshow('result', result)
cv2.waitKey(0)

print()

cv2.imwrite(os.path.join(base_dir, 'crosspoints.jpg'), result)
cv2.imwrite(os.path.join(base_dir, 'lines.jpg'), detectlines)
cv2.imwrite(os.path.join(base_dir, 'centers.jpg'), centerresult)