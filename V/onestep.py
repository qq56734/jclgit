import cv2
from skimage import morphology
import numpy as np
import os
import PIL
import imutils
from qumaoci2 import qumaoci


from getmask import get_mask

base_dir = 'data/raw'

filedirs = os.listdir(base_dir)


def isedge(point):
    numpoint = 0
    if point[1] == 720 or point[0] == 1280:
        return True
    for i in range(-1, 2):
        for j in range(-1, 2):
            if gray[point[1] + i][point[0] + j] == 255:
                numpoint += 1
    return numpoint >= 3


def findnext(pre, point):
    numpoints = 0
    t= 0

    if point[0] == 720 or point[1] == 1280:
        return point


    for i in range(-1, 2):
        for j in range(-1, 2):
            if gray[point[1] + j][point[0] + i] == 255:
                numpoints += 1
                x = point[0] + i
                y = point[1] + j
                #print((x, y) != (pre[0], pre[1]))
                if (x, y) != (pre[0], pre[1]) and (x, y) != (point[0], point[1]):
                    nextpoint = np.array([x, y])
                    t = 1
                    #print(nextpoint)
    if numpoints >= 4:
        return point

    if t== 0:
        return point
    return findnext(point, nextpoint)






for filename in filedirs:


    print(filename)

    filename = filename


    imgdir = os.path.join(base_dir, filename)

    img = PIL.Image.open(imgdir)

    img = img.resize((720, 540), PIL.Image.BILINEAR)

    out_dir = os.path.join('data', filename[:-4])

    os.makedirs(out_dir, exist_ok=True)

    img.save(os.path.join(out_dir, 'raw.jpg'))

    mask = get_mask(img)




    #
    #
    cv2.imwrite(os.path.join(out_dir, 'mask.png'), mask)
    #
    #



    mask[mask==255] = 1
    skeleton0 = morphology.skeletonize(mask)
    skeleton = skeleton0.astype(np.uint8)*255


    cv2.imwrite(os.path.join(out_dir, 'skeleton.png'), skeleton)

    #kernel = np.ones((7, 7), np.uint8)
    #mask = cv2.dilate(mask, kernel, iterations=2)

    #cv2.imwrite(os.path.join(out_dir, 'mask_dilate.png'), mask)




    #gray = cv2.imread(os.path.join(out_dir, 'mask.png'))
    #gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray = skeleton

    gray = qumaoci(gray)

    cv2.imwrite(os.path.join(out_dir, 'skeletonqumaoci.png'), gray)

    #cv2.RETR_EXTERNAL
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
    docCnt = None








    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                docCnt = approx
                break
    #img = cv2.imread(imgdir)

    imggray = gray

    # img = cv2.imread(imgdir)
    #
    # img = cv2.resize(img, (1008, 756))

    #img = cv2.flip(img,-1)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    for peak in docCnt:
        peak = peak[0]
        #if not isedge(peak):
            #peak = findnext(peak,peak)
        print(peak)
        cv2.circle(imggray, tuple(peak), 6, (255, 0, 0))
        cv2.circle(img, tuple(peak), 6, (255, 0, 0))


    cv2.imwrite(os.path.join(out_dir, '4pointm.jpg'), imggray)
    cv2.imwrite(os.path.join(out_dir, '4point.jpg'), img)


