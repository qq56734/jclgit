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



def findnext(gray1, point, shape):


    gray1[point[1]][point[0]] = 0

    numpoints = 0

    nextpoints =[]


    irange = [-1 , 0 ,1]

    jrange = [-1 , 0 , 1]


    if point[1] == shape[0] - 1 :
        jrange.remove(1)

    if point[0] == shape[1] - 1:
        irange.remove(1)

    if point[1] == 0 :
        jrange.remove(-1)

    if point[0] == 0:
        irange.remove(-1)





    for i in irange:
        for j in jrange:
            if gray1[point[1] + j][point[0] + i] == 255:
                numpoints += 1
                x = point[0] + i
                y = point[1] + j
                #print((x, y) != (pre[0], pre[1]))



                if i == 0 or j == 0:
                    nextpoints.insert(0, np.array([x, y]))
                else:
                    nextpoints.append(np.array([x, y]))
                    #print(nextpoint)

    #print(point)
    #print(numpoints)
    #print(nextpoints)
    if numpoints >= 2:
        #gray1[point[1]][point[0]] = 255
        return


    if len(nextpoints) > 0:
        findnext(gray1, nextpoints[0], shape)







import numpy as np

def qumaoci(gray):


    gray1 = gray.copy()



    print(gray.shape)

    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[1]


    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
    cnt = cnts[0]

    for point in cnt:

        if not isedge(gray, point[0], gray.shape):

            findnext(gray1, point[0], gray.shape)

            #cv2.circle(gray1, tuple(point[0]), 6, (255, 0, 0))





    return gray1


