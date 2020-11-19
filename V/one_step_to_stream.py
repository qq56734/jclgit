import cv2
import numpy as np
import os
import time
from fourpoint import get_4point
from PIL import Image



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

    # cv2.imshow('frame', frame)
    #
    # cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    copy = cv2.resize(frame.copy(),(640, 480))

    cv2.imwrite(filepath, copy)

    return frame


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






url = 'rtsp://admin:shijue666@10.1.2.111:554/h264/ch1/main/av_stream'
base_dir = 'data/stream'

out_dir = os.path.join(base_dir, str(time.time()))
os.makedirs(out_dir, exist_ok = True)


ball_color = 'blue'

color_dist = {'red': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([0, 255, 0])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 25]), 'Upper': np.array([90, 255, 255])},
              }



font = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3,3),np.uint8)



frame = cap_screen(url, out_dir)
#frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))



fourpoints = get_4point(frame, out_dir)
#skeleton= skeleton.astype('int64')


print(fourpoints)

longside,  fourpoints = get_rank(fourpoints)





img = cv2.imread(os.path.join(out_dir, 'frame.jpg'))

# direction = {0:(-1, -1), 1:(1, -1), 2:(-1,1), 3:(1, 1)}
# ran = 4
# for i in range (len(fourpoints)):
#     fourpoints[i] = refine(skeleton,fourpoints[i], direction[i], ran)
#     cv2.circle(img, tuple(fourpoints[i]), 6, (0, 0, 255))
# cv2.imwrite(os.path.join(out_dir, 'refined.jpg'), img)

#src = np.float32([[212, 76], [ 427,71], [214, 254], [432, 250]])


src = np.float32(fourpoints)

print(fourpoints)

print(longside)

longside = 1220

shortside = 1000

if longside:
    dst = np.float32([[0, 0], [longside , 0], [0, shortside ], [longside , shortside]])
else:
    dst = np.float32([[0, 0], [shortside, 0], [0, longside], [shortside, longside]])

m = cv2.getPerspectiveTransform(src, dst)

url = 'rtsp://admin:shijue666@10.1.2.111:554/h264/ch1/sub/av_stream'
cap = cv2.VideoCapture(url)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        if frame is not None:

            start = time.time()


            #frame = cv2.warpPerspective(frame, m, (1600, 670))
            #frame = cv2.resize(frame,(640, 360))

            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像


            #erode_hsv = hsv
            inRange_hsv = cv2.inRange(hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])

            #erode_hsv = cv2.erode(inRange_hsv, kernel, iterations=2)  # 腐蚀 粗的变细

            erode_hsv = inRange_hsv


            cnts = cv2.findContours(erode_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)

                # points = c
                #
                # print(points)
                #
                # points = np.insert(points, 2, 1, axis = 1)
                #
                # print(points)
                #
                #
                # points = np.dot(m, points.T)
                #
                # points = points.T
                #
                # print(points)
                #
                # points[:,0] = points[:,0]/points[:,2]
                # points[:,1] = points[:,1]/points[:,2]
                #
                # points = points[:, 0:1]
                #
                # print(points)




                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                x = (box[0][0] + box[2][0])/2
                y = (box[0][1] + box[2][1])/2
                point = np.array([x,y])
                print(point)
                point = np.insert(point, 2, 1, 0)
                new_point = np.dot(m, point.T)

                x = new_point[0]/new_point[2]
                y = new_point[1]/new_point[2]
                cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
                cv2.putText(frame, '{:.2f},{:.2f}'.format(x, y), (40, 40), font, 1, (255,0,255), 2)

            for point in fourpoints:

                cv2.circle(frame, tuple(point), 6, (255, 0, 0))





            cv2.imshow('camera', frame)
            end = time.time()

            seconds = end - start

            fps = 1 / (seconds + 1)

            print("Estimated frames per second : {0}".format(fps))

            cv2.waitKey(1)
        else:
            print("无画面")
    else:
        print("无法读取摄像头！")
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


