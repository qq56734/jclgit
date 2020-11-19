import cv2
def isedge(gray, point, shape):



    if point[1] == shape[0] - 1 or point[0] == shape[1] - 1 or point[1] == 0 or point[0] == 0:
        return False

    numpoint = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if gray[point[1] + i][point[0] + j] == 255:
                numpoint += 1
    return numpoint >= 3



def findnext(gray1, point, shape, maodians, previous):

    # gray0 = gray1.copy()
    # cv2.circle(gray0, tuple(point), 6, (255, 0, 0))
    # cv2.imshow('point', gray0)
    # cv2.waitKey(0)

    maodians.append(point)
    #gray1[point[1]][point[0]] = 0

    numpoints = 0

    nextpoints =[]


    # irange = [-1 , 0 ,1]
    #
    # jrange = [-1 , 0 , 1]
    #
    #
    # if point[1] == shape[0] - 1 :
    #     jrange.remove(1)
    #
    # if point[0] == shape[1] - 1:
    #     irange.remove(1)
    #
    # if point[1] == 0 :
    #     jrange.remove(-1)
    #
    # if point[0] == 0:
    #     irange.remove(-1)

    steps = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for step in steps:
        i, j = step

        if gray1[point[1] + j][point[0] + i] == 255:
            if step != previous:
                numpoints += 1
                x = point[0] + i
                y = point[1] + j
            # print((x, y) != (pre[0], pre[1]))
                nextpoints.append(np.array([x, y]))



    # for i in irange:
    #     for j in jrange:
    #         if gray1[point[1] + j][point[0] + i] == 255:
    #             numpoints += 1
    #             x = point[0] + i
    #             y = point[1] + j
    #             #print((x, y) != (pre[0], pre[1]))
    #
    #
    #
    #             if x != previous[0] or y != previous[1]:
    #                 if i == 0 or j == 0:
    #                     nextpoints.insert(0, np.array([x, y]))
    #                 else:
    #                     nextpoints.append(np.array([x, y]))
    #                     #print(nextpoint)

    #print(point)
    #print(numpoints)
    #print(nextpoints)
    if numpoints >= 2 or len(maodians) > 30:
        if numpoints == 2:
            maodians.pop(-1)
        #gray1[point[1]][point[0]] = 255
        return




    if len(nextpoints) > 0:


        nextpoint = nextpoints[0]

        #print(nextpoint)

        previous = (point[0]- nextpoint[0], point[1] - nextpoint[1])

        findnext(gray1, nextpoint, shape, maodians, previous)







import numpy as np

def qumaoci(gray, frame):


    gray1 = gray.copy()



    print(gray.shape)


    distance = int((gray.shape[0] + gray.shape[1])/(640 + 480)*30)

    print('distance: {}'.format(distance))

    gray_center = np.array((gray.shape[1]/2, gray.shape[0]/2))

    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[1]

    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, False), reverse=True)  # 根据轮廓面积从大到小排序


    dis = 10000000

    for i in range(3):

        cnt_s = np.squeeze(cnts[i])

        center_p = cnt_s.mean(axis=0)

        new_dis = np.linalg.norm(center_p - gray_center)

        if new_dis < dis :
            # print(cnt)
            dis  = new_dis
            cnt = cnts[i]

    copy = frame.copy()
    cv2.drawContours(copy,[cnt],0,(0,0,255),3)
    cv2.imshow('contour',copy)
    cv2.waitKey(0)

    duandians = []


    for point in cnt:

        if not isedge(gray, point[0], gray.shape):

            #print(point)

            #cv2.circle(gray1, tuple(point[0]), 1, (255, 0, 0))

            #cv2.imwrite('gray.png', gray1)

            maodians = []



            findnext(gray1, point[0], gray.shape, maodians, (0, 0))

            #cv2.circle(gray1, tuple(point[0]), 6, (255, 0, 0))
            #print(len(maodians))

            if len(maodians) < distance:

                for maodian in maodians:
                    gray1[maodian[1]][maodian[0]] = 0
            else:
                duandians.append(point[0])
    print('duandian: {}'.format(duandians))
    if len(duandians) == 2:
        cv2.line(gray1, tuple(duandians[0]), tuple(duandians[1]), (255, 0, 0))

    return gray1


