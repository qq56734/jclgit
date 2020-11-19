import cv2
import numpy as np
import time


# def left(angle):
#     GPIO.output(IN1, GPIO.LOW)
#     GPIO.output(IN2, GPIO.LOW)
#     GPIO.output(IN3, GPIO.HIGH)
#     GPIO.output(IN4, GPIO.LOW)
#     pwm_ENA.ChangeDutyCycle(10)
#     pwm_ENB.ChangeDutyCycle(10)
#
#     time.sleep(angle/90*0.81)
#
# def right(angle):
#     GPIO.output(IN1, GPIO.HIGH)
#     GPIO.output(IN2, GPIO.LOW)
#     GPIO.output(IN3, GPIO.LOW)
#     GPIO.output(IN4, GPIO.HIGH)
#     pwm_ENA.ChangeDutyCycle(10)
#     pwm_ENB.ChangeDutyCycle(10)
#
#     time.sleep(angle / 90 * 0.81)
#
# def forward(distance):
#
#
#     GPIO.output(IN1, GPIO.HIGH)
#     GPIO.output(IN2, GPIO.LOW)
#     GPIO.output(IN3, GPIO.HIGH)
#     GPIO.output(IN4, GPIO.LOW)
#     pwm_ENA.ChangeDutyCycle(20)
#     pwm_ENB.ChangeDutyCycle(20)
#     time.sleep(distance/290)
#
#
#
#
#
# def run_to_target(angle, distance):
#     if angle > 0:
#         left(angle)
#     else:
#         right(abs(angle))
#
#     forward(distance)


def inrange(loc1, loc2):
    return np.linalg.norm(loc1 - loc2) < 50


def get_command(poss, target):
    direction0 = poss[0] - poss[1]

    pos = (poss[0] + poss[1]) / 2

    direction = target - pos

    distance = np.linalg.norm(direction)

    cosangle = direction.dot(direction0) / (np.linalg.norm(direction) * np.linalg.norm(direction0))

    angle = np.arccos(cosangle) * 360 / 2 / np.pi

    if direction0[0] * direction[1] - direction[0] * direction0[1] < 0:
        angle = - angle

    return angle, distance


def get_pos(cap):
    poss = []

    ret, frame = cap.read()

    while frame is None:

        print('capturing')

        ret, frame = cap.read()


# frame = cv2.warpPerspective(frame, m, (1600, 670))
# frame = cv2.resize(frame,(640, 360))

    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像

    colors = ['red', 'green']

    for i in range(2):

        color = colors[i]

        # erode_hsv = hsv
        inRange_hsv = cv2.inRange(hsv, color_dist[color]['Lower'], color_dist[color]['Upper'])

        # erode_hsv = cv2.erode(inRange_hsv, kernel, iterations=2)  # 腐蚀 粗的变细

        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

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
            x = (box[0][0] + box[2][0]) / 2
            y = (box[0][1] + box[2][1]) / 2
            point = np.array([x, y])
            point = np.insert(point, 2, 1, 0)
            new_point = np.dot(m, point.T)
            x = new_point[0] / new_point[2]
            y = new_point[1] / new_point[2]

            poss.append(np.array([x, y]))

            cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
    cap.release()
    return frame, poss

# import RPi.GPIO as GPIO
# import time
#
# #小车电机引脚定义
# IN1 = 20
# IN2 = 21
# IN3 = 19
# IN4 = 26
# ENA = 16
# ENB = 13
#
# #设置GPIO口为BCM编码方式
# GPIO.setmode(GPIO.BCM)
#
# #忽略警告信息
# GPIO.setwarnings(False)
#
# #电机引脚初始化操作
# def motor_init():
#     global pwm_ENA
#     global pwm_ENB
#     global delaytime
#     GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
#     GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
#     GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
#     GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)
#     GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
#     GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
#     #设置pwm引脚和频率为2000hz
#     pwm_ENA = GPIO.PWM(ENA, 2000)
#     pwm_ENB = GPIO.PWM(ENB, 2000)
#     pwm_ENA.start(0)
#     pwm_ENB.start(0)


url = 'rtsp://admin:shijue666@10.1.2.111:554/h264/ch1/main/av_stream'

color_dist = {'red': {'Lower': np.array([156, 43, 46]), 'Upper': np.array([180, 255, 255])},
              'blue': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 50]), 'Upper': np.array([90, 255, 255])},
              }

font = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3, 3), np.uint8)

src = np.float32([[489, 203.7], [932.6, 192.8], [504.8, 580.9], [944.2, 549.6]])
dst = np.float32([[0, 0], [1220, 0], [0, 1000], [1000, 1220]])
m = cv2.getPerspectiveTransform(src, dst)

points = np.array([[200, 100]])

for i in range(len(points)):

    cap = cv2.VideoCapture(url)

    frame, poss = get_pos(cap)

    poss = np.array([[10, 20], [10, 30]])

    loc = (poss[0] + poss[1]) / 2

    cv2.imwrite('../deteciton.jpg', frame)

    print('head: {},{}, tail: {},{}'.format(poss[0][1], poss[0][1], poss[1][1], poss[1][1]))

    while not inrange(points[i], loc):
        print('directing to point {},{}\n'.format(points[i][0], points[i][1]))

        angle, distance = get_command(poss, points[i])

        print('angle: {}, distance: {}\n'.format(angle, distance))

        # run_to_target(angle, distance)
        # time.sleep(1)

        cap = cv2.VideoCapture(url)

        frame, poss = get_pos(cap)

        poss = np.array([[10, 20], [10, 30]])

        print('head: {},{}, tail: {},{}'.format(poss[0][1], poss[0][1], poss[1][1], poss[1][1]))

        cv2.imwrite('test/' + str(time.time()) + '.jpg', frame)

        loc = (poss[0] + poss[1]) / 2

        time.sleep(1)



