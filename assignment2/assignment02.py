import cv2
import robomaster
from robomaster import robot, camera
import time
import numpy as np


def sub_position_handler(position_info):
    global x
    x, y, z = position_info
    
    print("position: x:{0}, y:{1}, z:{2}".format(x, y, z))


def detect_chicken():
    max_chick_contour = max(chick_contours, key=cv2.contourArea)
    (x_chick, y_chick, w_chick, h_chick) = cv2.boundingRect(max_chick_contour)

    if w_chick > 65:
        adj_w_chick = int(w_chick * 0.3)
        adj_h_chick = int(h_chick * 0.6)
        new_y_chick = y_chick - adj_h_chick // 2 + 20
    else:
        adj_w_chick = int(w_chick * 0.3)
        adj_h_chick = int(h_chick * 0.5)
        new_y_chick = (y_chick - adj_h_chick // 2 + 20) - 18

    new_x_chick = x_chick - adj_w_chick // 2
    new_w_chick = w_chick + adj_w_chick
    new_h_chick = h_chick + adj_h_chick

    cv2.rectangle(frame, (new_x_chick, new_y_chick), (new_x_chick + new_w_chick, new_y_chick + new_h_chick), (255, 0, 255), 2)
    cv2.putText(frame, f" Chicken ", (x_chick, y_chick - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)



def detect_bottle():
    max_bottle_contour = max(bottle_contours, key=cv2.contourArea)
    (x_bottle, y_bottle, w_bottle, h_bottle) = cv2.boundingRect(max_bottle_contour)

    if w_bottle > 85:   
        adj_w_bottle = int(w_bottle * 0.27)  
        adj_h_bottle = int(h_bottle * 2.5)
        new_y_bottle = y_bottle - adj_h_bottle // 2 -10

    elif w_bottle > 50:  
        adj_w_bottle = int(w_bottle * 0.27)  
        adj_h_bottle = int(h_bottle * 2.9)
        new_y_bottle = y_bottle - adj_h_bottle // 2 -11    

    elif w_bottle > 35:  
        adj_w_bottle = int(w_bottle * 0.28)  
        adj_h_bottle = int(h_bottle * 3.4)
        new_y_bottle = y_bottle - adj_h_bottle // 2 -8

    else:
        adj_w_bottle = int(w_bottle * 0.47)
        adj_h_bottle = int(h_bottle * 3.4)
        new_y_bottle = y_bottle - adj_h_bottle // 2 -3

    new_x_bottle = x_bottle - adj_w_bottle // 2
    new_w_bottle = w_bottle + adj_w_bottle
    new_h_bottle = h_bottle + adj_h_bottle

    cv2.rectangle(frame, (new_x_bottle, new_y_bottle), (new_x_bottle + new_w_bottle, new_y_bottle + new_h_bottle), (0, 165, 255), 2)
    cv2.putText(frame, f" Bottle ", (x_bottle, y_bottle-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)


if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis

    center_x = 1280 / 2
    center_y = 720 / 2

    x = 0.0  
    target_distance = 1.2  # เป้าหมายระยะทางที่ต้องการ

    ep_chassis.sub_position(freq=10, callback=sub_position_handler)
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    ep_gimbal.moveto(pitch=-10, yaw=0).wait_for_completed()

    while True:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_chick = np.array([33, 150, 100])
        upper_chick = np.array([36, 255, 255])

        lower_bottle = np.array([95, 80, 100])
        upper_bottle = np.array([120, 255, 255])

        mask_chick = cv2.inRange(hsv_frame, lower_chick, upper_chick)
        mask_bottle = cv2.inRange(hsv_frame, lower_bottle, upper_bottle)
        chick_contours, _ = cv2.findContours(
            mask_chick.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bottle_contours, _ = cv2.findContours(
            mask_bottle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if chick_contours: 
            detect_chicken()
        if bottle_contours:
            detect_bottle()

        # การเดินแบบปกติไม่ใช้ PID
        if target_distance > x:
            speed = 15  # ความเร็วที่ตั้งไว้ล่วงหน้า
            ep_chassis.drive_wheels(w1=speed, w2=speed, w3=speed, w4=speed)
            time.sleep(0.005)
        else:
            ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            time.sleep(0.005)
        cv2.imshow("Original Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)

    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_chassis.unsub_position()
    ep_robot.close()