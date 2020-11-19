import cv2
from skimage import morphology
import numpy as np
import os
import PIL
import imutils
from tools.qumaoci2 import qumaoci
import time

from tools.getmask import get_mask




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








def cap_screen(url, out_dir):


    filepath = os.path.join(out_dir, 'frame.jpg')



    cap = cv2.VideoCapture(url)
    ret,frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    # cv2.imshow('frame', frame)
    #
    # cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    cv2.imwrite(filepath, frame)

    return frame


def get_4point_raw(frame, out_dir):




    mask = get_mask(frame)

    cv2.imwrite(os.path.join(out_dir, 'mask.png'), mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    cv2.imwrite(os.path.join(out_dir, 'mask_dilate.png'), mask)


    mask[mask == 255] = 1
    skeleton0 = morphology.skeletonize(mask)
    skeleton = skeleton0.astype(np.uint8) * 255

    cv2.imwrite(os.path.join(out_dir, 'skeleton.png'), skeleton)



    # gray = cv2.imread(os.path.join(out_dir, 'mask.png'))
    # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray = skeleton

    gray = qumaoci(gray)

    #gray = qumaoci(gray)

    cv2.imwrite(os.path.join(out_dir, 'skeletonqumaoci.png'), gray)

    # cv2.RETR_EXTERNAL
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                docCnt = approx
                break
    # img = cv2.imread(imgdir)



    #img = cv2.resize(img, (756, 1008))

    #img = cv2.flip(img, -1)

    points = np.zeros((4, 2), dtype=int)

    i = 0

    for peak in docCnt:
        peak = peak[0]
        points[i] = peak
        cv2.circle(gray, tuple(peak), 6, (255, 0, 0))
        cv2.circle(frame, tuple(peak), 6, (255, 0, 0))

        i += 1


    cv2.imwrite(os.path.join(out_dir, '4pointm.jpg'), gray)
    cv2.imwrite(os.path.join(out_dir, '4point.jpg'), frame)

    return points, gray

def refine(edges, point, direction, ran, maxextend = 5):

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



def get_4point(longside, shortside):

    url = 'rtsp://admin:shijue666@10.1.2.111:554/h264/ch1/main/av_stream'
    base_dir = 'data/stream'

    out_dir = os.path.join(base_dir, str(time.time()))
    os.makedirs(out_dir, exist_ok = True)




    font = cv2.FONT_HERSHEY_SIMPLEX
    kernel = np.ones((3,3),np.uint8)



    frame = cap_screen(url, out_dir)

    fourpoints, skeleton = get_4point_raw(frame, out_dir)

    skeleton= skeleton.astype('int64')


    print('fourpoints: \n{}'.format(fourpoints))

    iflongside,  fourpoints = get_rank(fourpoints)


    direction = {0:(-1, -1), 1:(1, -1), 2:(-1,1), 3:(1, 1)}
    ran = 4


    img = cv2.imread(os.path.join(out_dir, 'frame.jpg'))


    for i in range (len(fourpoints)):
        fourpoints[i] = refine(skeleton,fourpoints[i], direction[i], ran)
        cv2.circle(img, tuple(fourpoints[i]), 6, (0, 0, 255))

    cv2.imwrite(os.path.join(out_dir, 'refined.jpg'), img)

    #src = np.float32([[212, 76], [ 427,71], [214, 254], [432, 250]])

    #fourpoints.astype('int')






    print('fourpoints in rank: \n{}'.format(fourpoints))



    src = np.float32(fourpoints)



    if iflongside:
        dst = np.float32([[0, 0], [longside , 0], [0, shortside ], [longside , shortside]])
    else:
        dst = np.float32([[0, 0], [shortside, 0], [0, longside], [shortside, longside]])


    print('dst: \n{}'.format(dst))

    return src, dst





