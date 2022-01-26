# -*- coding: UTF-8 -*-
"""
@File    ：AIVirtualMouseProject.py
@Author  ：夏泽城
@Time    :2021-09-21  16:56
@CSDN    ：https://blog.csdn.net/py5326?spm=1000.2115.3001.5113
"""
import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # 帧缩减
smoothening = 7  # 平滑度
########################

pTime = 0
plocX, plocY = 0, 0  # 上一个坐标，x，y
clocX, clocY = 0, 0  # 当前坐标，x,y

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. 找到手指的Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. 得到食指与中指的坐标
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. 检测哪根手指是向上的
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        # 4. 只有手指朝上的时候事移动模式
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. 移动的时候进行坐标转换，以保证鼠标移动到屏幕的正确位置
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. 平滑移动的值smoothvalue
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. 移动鼠标
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. 当两根手指都朝上的时候处于点击模式
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. 得到两根手指的距离
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. 当两根手指距离很短的时候判断为点击
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. 设置帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. 显示
    cv2.imshow("Image", img)
    cv2.waitKey(1)
