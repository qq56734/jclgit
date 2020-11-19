import cv2
import numpy as np
import time



url = 'rtsp://admin:shijue666@10.1.2.111:554/h264/ch1/sub/av_stream'
cap = cv2.VideoCapture(url)


ball_color = 'green'

color_dist = {'red': {'Lower': np.array([156, 43, 46]), 'Upper': np.array([180, 255, 255])},
              'blue': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 50]), 'Upper': np.array([90, 255, 255])},
              }



font = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3,3),np.uint8)

#src = np.float32([[248.5, 173.7], [ 470.3,129.4], [281.3, 422.2], [492, 363.1]])
#dst = np.float32([[0, 0], [1220 , 0], [0, 1000 ], [1220 , 1000]])


src = np.float32([[125.2, 171.4], [ 534,79], [178, 444.9], [566.6, 329.5]])
dst = np.float32([[0, 0], [2400 , 0], [0, 1200 ], [2400 , 1200]])



m = cv2.getPerspectiveTransform(src, dst)
print(m)
#src = np.insert(src, 2, 1, axis = 1)
#print(src)
#print(np.dot( m, src.T))
#print(np.dot( m, np.array([694, 638, 1]).T))

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

                # M = cv2.moments(c)  # 计算第一条轮廓的各阶矩,字典形式
                # center_x = M["m10"] / M["m00"]
                # center_y = M["m01"] / M["m00"]
                #
                # new_point = np.array([center_x, center_y, 1])
                #
                # new_point
                # = np.dot(m, new_point.T)
                #
                # x = new_point[0] / new_point[2]
                # y = new_point[1]/new_point[2]
                #
                # cv2.drawContours(frame, c, -1, 255, -1)  # 绘制轮廓，填充
                # cv2.circle(frame, (int(center_x), int(center_y)), 2, 128, -1)  # 绘制中心点




                cv2.putText(frame, '{:.2f},{:.2f}'.format(x, y), (40, 40), font, 1, (255,0,255), 2)
            cv2.imshow('camera', frame)
            end = time.time()

            seconds = end - start

            fps = 1 / (seconds )

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