import robomaster
from robomaster import robot, blaster, camera
import math
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import pandas as pd

MAX_SPEED = 10
sapeed_Z_na = 200
#0 = front wall from north
#1 = left wall from north
#2 = right wall from north
#3 = back wall from north
#4 = visit
#[North(front), West(left), East(right), South(back),Visited,chicken,thive]
grid = [
    [[2,2,0,0,0,0,0],[2,0,0,0,0,0,0],[2,0,0,0,0,0,0],[2,0,0,0,0,0,0],[2,0,0,0,0,0,0],[2,0,2,0,0,0,0]],
    [[0,2,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,2,0,0,0,0]],
    [[0,2,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,2,0,0,0,0]],
    [[0,2,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,2,0,0,0,0]],
    [[0,2,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,2,0,0,0,0]],
    [[0,2,0,2,0,0,0],[0,0,0,2,0,0,0],[0,0,0,2,0,0,0],[0,0,0,2,0,0,0],[0,0,0,2,0,0,0],[0,0,2,2,0,0,0]]
]

# start map
row = 5
col = 2
cerrent_po = [row,col]
list_travel = []
direction_robot = []
list_chick = []
list_acrylic = []
found_acrylic = False
found_chick = False

# Function to save list_travel to CSV, including the start column
def save_travel_data(direction_robot, list_travel, filename="list_path.csv"):
    # Create DataFrame without repeating start_position
    df_travel = pd.DataFrame({"direction": direction_robot, "travel": list_travel})
    # Save the travel data to CSV
    df_travel.to_csv(filename, index=False)
    # Add the start_position to the first row manually

def save_data_acrylic(list_acrylic, filename="list_acrylic.csv"):
    df_acrylic = pd.DataFrame({"acrylic": list_acrylic})
    df_acrylic.to_csv(filename, index=False)

def save_data_chicken(list_chick, filename="list_chicken.csv"):
    df_chicken = pd.DataFrame({"chicken": list_chick})
    df_chicken.to_csv(filename, index=False)


''' ----- image ----- '''
class Marker:
    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    @property
    def pt1(self):
        return int(self._x), int(self._y)

    @property
    def pt2(self):
        return int(self._x + self._w), int(self._y + self._h)

    @property
    def center(self):
        return int(self._x + self._w / 2), int(self._y + self._h / 2)


def detect_blue_circles(frame, count):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50, 
        param1=50, 
        param2=25, 
        minRadius=10, 
        maxRadius=50
    )

    marker = None

    if circles is not None:
        print('พบวงกลม')
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]

            roi = mask[y - r:y + r, x - r:x + r]
            if roi.size == 0:
                continue

            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity > 0.65 :
                    cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

                    top_left = (x - r, y - r)
                    bottom_right = (x + r, y + r)

                    top_left = (max(top_left[0], 0), max(top_left[1], 0))
                    bottom_right = (min(bottom_right[0], frame.shape[1] - 1), min(bottom_right[1], frame.shape[0] - 1))

                    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

                    w = bottom_right[0] - top_left[0]
                    h = bottom_right[1] - top_left[1]

                    x,y,w,h = top_left[0]-15, top_left[1]-5, w+27, h+270
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f" x: {x}, y: {y}, w: {w}, h: {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    marker = Marker(x, y, w, h)
                    break 
            if marker:
                break
    
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)  # แนวนอน
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)  # แนวตั้ง

    return marker


def detect():
    global count, check, accumulate_err_x ,accumulate_err_y,  prev_time, found_acrylic

    for _ in range(3):
        time.sleep(0.5)
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)[300:650,200:1070]
    cv2.imshow('Frame', frame)
    cv2.waitKey(1) 
    while True:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)[300:650,200:1070]
        marker = detect_blue_circles(frame, count)
        detect_chicken(frame)
        if marker and marker._w >= 45 and marker._w < 120 and count == 0: check = False
        if count > 0:count += 1
        
        print('--->',check)
        if check: break

        if check == False:
            ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            time.sleep(0.1)
            if marker:
                after_time = time.time()
                x, y = marker.center

                err_x = center_x - x
                err_y = center_y - y
                accumulate_err_x += err_x * (after_time - prev_time)
                accumulate_err_y += err_y * (after_time - prev_time)

                speed_x = p * err_x
                speed_y = p * err_y
                ep_gimbal.drive_speed(pitch_speed=speed_y, yaw_speed=-speed_x)

                prev_time = after_time

                if marker._w >= 45 and count == 0: count += 1
                time.sleep(0.1)
                    

            if count == 13:
                ep_blaster.set_led(brightness=200, effect=blaster.LED_ON)
                time.sleep(0.1)

            if count == 17 :
                ep_blaster.fire(times=5)
                print('shouttttttttttttttttttttttttttttttttttt')
                found_acrylic = True

            elif count >=25:
                ep_gimbal.recenter(pitch_speed=400, yaw_speed=400).wait_for_completed()
                ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
                check, count = True, 0
                print('---------')
                break
            else:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)  



def detect_chicken(frame):
    global found_chick
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_chicken = np.array([33, 150, 100])
    upper_chicken = np.array([38, 255, 255])

    mask_chicken = cv2.inRange(hsv_frame, lower_chicken, upper_chicken)
    contours_chicken, _ = cv2.findContours(mask_chicken, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_chicken: 
        max_chicken_contour = max(contours_chicken, key=cv2.contourArea)
        (x_chicken, y_chicken, w_chicken, h_chicken) = cv2.boundingRect(max_chicken_contour)
        
        if  w_chicken > 30 and h_chicken > 50: 
            if w_chicken > 65:
                add_w_chicken = int(w_chicken * 0.3)
                add_h_chicken = int(h_chicken * 0.6)
                new_y_chicken = y_chicken - add_h_chicken // 2 + 20

            elif w_chicken > 30 :
                add_w_chicken = int(w_chicken * 0.35)
                add_h_chicken = int(h_chicken * 0.55)
                new_y_chicken = (y_chicken - add_h_chicken // 2 + 20) - 15

                found_chick = True
                print('chickkkkkkkkkkkk')

        else:
            add_w_chicken = int(w_chicken * 0.35)
            add_h_chicken = int(h_chicken * 0.55)
            new_y_chicken = (y_chicken - add_h_chicken // 2 + 20) - 15
        

        new_x_chicken = x_chicken - add_w_chicken // 2
        new_w_chicken = w_chicken + add_w_chicken
        new_h_chicken = h_chicken + add_h_chicken
        # found_chick = True
        # print('chickkkkkkkkkkkk')

        cv2.rectangle(frame, (new_x_chicken, new_y_chicken), (new_x_chicken + new_w_chicken, new_y_chicken + new_h_chicken), (255, 0, 255), 2)
        cv2.putText(frame, f" chicken x: {x_chicken}, y: {y_chicken}, w: {w_chicken}, h: {h_chicken}", (x_chicken, y_chicken - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)




''' ----- TOF Sensor ----- '''
status_tof = None
tof_distance = 0
def tof_data_handler(sub_info):
    global tof_distance, status_tof
    tof_distance = sub_info[0]
    sharp_sen_data()


''' ----- Sharp Sensor ----- '''
# status_ss_l = None
adc_l = None
adc_l_new = 0

# status_ss_r = None
adc_r = None
adc_r_new = 0
def sharp_sen_data():
    global adc_r,adc_l,adc_r_new,adc_l_new,status_ss_l,status_ss_r
    adc_r = ep_sensor_adaptor.get_adc(id=2, port=2)
    adc_r_cm = (adc_r * 3) / 1023  # process to cm unit
    adc_l = ep_sensor_adaptor.get_adc(id=3, port=1)
    adc_l_cm = (adc_l * 3) / 1023  # process to cm unit

    if adc_r_cm > 1.4:
        adc_r_new = ((adc_r_cm - 4.2) / -0.31)
    elif 1.4 >= adc_r_cm >= 0.6:
        adc_r_new = ((adc_r_cm - 2.03) / -0.07)
    elif 0 <= adc_r_cm < 0.6:
        adc_r_new = ((adc_r_cm - 0.95) / -0.016)

    if adc_l_cm > 1.4:
        adc_l_new = ((adc_l_cm - 4.2) / -0.31)
    elif 1.4 >= adc_l_cm >= 0.6:
        adc_l_new = ((adc_l_cm - 2.03) / -0.07)
    elif 0 <= adc_l_cm < 0.6:
        adc_l_new = ((adc_l_cm - 0.95) / -0.016)
    
    # if 46 > adc_r_new > 2:
    #     status_ss_r = True
    # else:
    #     status_ss_r = False

    # if 40 > adc_l_new > 2:
    #     status_ss_l = True
    # else:
    #     status_ss_l = False

''' ----- sub_position_handler ----- '''
x = 0
y = 0
# ฟังก์ชันสำหรับจัดการตำแหน่งของหุ่นยนต์
def sub_position_handler(position_info):
    global x,y
    x, y, z = position_info


''' ----- sub_attitude_info_handler ----- '''
yaw = 0
threshold_sharp = 30

# ฟังก์ชันสำหรับจัดการท่าทางของหุ่นยนต์
def sub_attitude_info_handler(attitude_info):
    global yaw
    yaw, pitch, roll = attitude_info
    # print(f"chassis attitude: yaw:{yaw}")


''' ----- adjust_all_walls ----- '''
def adjust_wall():
    adjust_wall_f()
    adjust_wall_l()
    adjust_wall_r()

def adjust_wall_l():

    if adc_l_new < 40:
        if adc_l_new < 15:
            walk_y = (abs(15 - adc_l_new)/100)+0.0035
            # print("Move right")
            ep_chassis.move(x=0, y=walk_y, z=0, xy_speed=MAX_SPEED).wait_for_completed()
        elif adc_l_new > 35:
            walk_y = (abs(35 - adc_l_new)/100)+0.0035
            # print("Move left")
            ep_chassis.move(x=0, y=-walk_y, z=0, xy_speed=MAX_SPEED).wait_for_completed()

def adjust_wall_r():

    if adc_r_new < 40:
        if adc_r_new < 15 :
            walk_y = (abs(15 - adc_r_new)/100)+0.002
            # print("Move left")
            ep_chassis.move(x=0, y=-walk_y, z=0, xy_speed=MAX_SPEED).wait_for_completed()
        
        elif adc_r_new > 35:
            walk_y = (abs(35 - adc_r_new)/100)+0.002
            # print("Move right")
            ep_chassis.move(x=0, y=walk_y, z=0, xy_speed=MAX_SPEED).wait_for_completed()

def adjust_wall_f():    
    
    if tof_distance <200:
        walk_x = (abs(200-tof_distance)/1000)+0.0035
        ep_chassis.move(x=-walk_x, y=0, z=0, xy_speed=MAX_SPEED).wait_for_completed()
    
    if 520>=tof_distance>350:
        walk_x = (abs(tof_distance -350)/1000)+0.0015
        ep_chassis.move(x=walk_x, y=0, z=0, xy_speed=MAX_SPEED).wait_for_completed()
    
def adjust_angle():
    target_yaw = 0
    current_yaw = yaw

    if -135 < yaw <= -45:
        target_yaw = -90
        ep_chassis.move(x=0, y=0, z=current_yaw-target_yaw, z_speed=60).wait_for_completed()
    elif 45 < yaw < 135:
        target_yaw = 90
        ep_chassis.move(x=0, y=0, z=current_yaw-target_yaw, z_speed=60).wait_for_completed()
    elif -45 < yaw <= 45:
        target_yaw = 0
        ep_chassis.move(x=0, y=0, z=current_yaw, z_speed=60).wait_for_completed()
    elif -180 <= yaw < -135 :
        target_yaw = -180
        ep_chassis.move(x=0, y=0, z=current_yaw-target_yaw, z_speed=60).wait_for_completed() 
    elif 135 < yaw <= 180:
        target_yaw = 180
        ep_chassis.move(x=0, y=0, z=current_yaw-target_yaw, z_speed=60).wait_for_completed()


''' ----- movement ----- '''
def move_stop():
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.75)
    time.sleep(0.2)

def move_forward():
    print("Drive forward")
    ep_chassis.move(x=0.57, y=0, z=0,xy_speed=MAX_SPEED).wait_for_completed()
    

def turn_back():
    print("Turn Back")
    
    ep_chassis.move(x=0, y=0, z=180, z_speed=sapeed_Z_na).wait_for_completed()
    ep_gimbal.recenter(pitch_speed=400, yaw_speed=400).wait_for_completed()
    

def turn_left():
    print("Turn Left")
    
    ep_chassis.move(x=0, y=0, z=90,z_speed=sapeed_Z_na).wait_for_completed()
    ep_gimbal.recenter(pitch_speed=400, yaw_speed=400).wait_for_completed()
    

def turn_right():
    print('Turn Right')
    
    ep_chassis.move(x=0, y=0, z=-90, z_speed=sapeed_Z_na).wait_for_completed()
    ep_gimbal.recenter(pitch_speed=400, yaw_speed=400).wait_for_completed()
    
    
''' ----- DirectionFacing ----- '''
robo_status_now = None
def getDirectionFacing():
    global robo_status_now
    degrees = yaw
    if -45 <= degrees < 0 or 45>=degrees >= 0:
        robo_status_now = 'N'
    if 45 < degrees <= 135:
        robo_status_now = 'E'
    if 135 < degrees <=180 or -180<= degrees <-135 :
        robo_status_now = 'S'
    if -135 <= degrees < -45:
        robo_status_now = 'W'

visit_counts = [[0 for _ in range(6)] for _ in range(6)]
''' ----- Update_Maze ----- '''
status_logic = None
def update_wall():
    global cerrent_po,status_logic
    list_travel.append(tuple(cerrent_po))

    getDirectionFacing()
    direction_robot.append(robo_status_now)
    row, col = cerrent_po
    visit_counts[row][col] += 1
    logic()
    if robo_status_now =='N':
        if status_logic == 'move_forward':
            grid[cerrent_po[0]-1][cerrent_po[1]][4] = 1

            cerrent_po = [cerrent_po[0]-1,cerrent_po[1]]
    
        if status_logic == 'turn right': #turn right and move forward
            grid[cerrent_po[0]][cerrent_po[1]+1][4] = 1
            
            cerrent_po = [cerrent_po[0],cerrent_po[1]+1]
        
        if status_logic == 'turn back':
            grid[cerrent_po[0]+1][cerrent_po[1]][4] = 1

            cerrent_po = [cerrent_po[0]+1,cerrent_po[1]]

        if status_logic == 'turn left':
            grid[cerrent_po[0]][cerrent_po[1]-1][4] = 1

            cerrent_po = [cerrent_po[0],cerrent_po[1]-1]       

    if robo_status_now =='E':  
        if status_logic == 'move_forward':
            grid[cerrent_po[0]][cerrent_po[1]+1][4] = 1

            cerrent_po = [cerrent_po[0],cerrent_po[1]+1]
    
        if status_logic == 'turn right': #turn right and move forward
            grid[cerrent_po[0]+1][cerrent_po[1]][4] = 1
            
            cerrent_po = [cerrent_po[0]+1,cerrent_po[1]]
        
        if status_logic == 'turn back':
            grid[cerrent_po[0]][cerrent_po[1]-1][4] = 1

            cerrent_po = [cerrent_po[0],cerrent_po[1]-1]

        if status_logic == 'turn left':
            grid[cerrent_po[0]-1][cerrent_po[1]][4] = 1

            cerrent_po = [cerrent_po[0]-1,cerrent_po[1]]

    if robo_status_now =='S':
        if status_logic == 'move_forward':
            grid[cerrent_po[0]+1][cerrent_po[1]][4] = 1


            cerrent_po = [cerrent_po[0]+1,cerrent_po[1]]
    
        if status_logic == 'turn right': #turn right and move forward
            grid[cerrent_po[0]][cerrent_po[1]-1][4] = 1
            
            cerrent_po = [cerrent_po[0],cerrent_po[1]-1]
        
        if status_logic == 'turn back':
            grid[cerrent_po[0]+1][cerrent_po[1]][4] = 1

            cerrent_po = [cerrent_po[0]+1,cerrent_po[1]]

        if status_logic == 'turn left':
            grid[cerrent_po[0]][cerrent_po[1]+1][4] = 1
            cerrent_po = [cerrent_po[0],cerrent_po[1]+1]      

    if robo_status_now =='W':
        if status_logic == 'move_forward':
            grid[cerrent_po[0]][cerrent_po[1]-1][4] = 1

            cerrent_po = [cerrent_po[0],cerrent_po[1]-1]
    
        if status_logic == 'turn right': #turn right and move forward
            grid[cerrent_po[0]-1][cerrent_po[1]][4] = 1
            
            cerrent_po = [cerrent_po[0]-1,cerrent_po[1]]
        
        if status_logic == 'turn back':
            grid[cerrent_po[0]][cerrent_po[1]+1][4] = 1

            cerrent_po = [cerrent_po[0],cerrent_po[1]+1]

        if status_logic == 'turn left':
            grid[cerrent_po[0]+1][cerrent_po[1]][4] = 1

            cerrent_po = [cerrent_po[0]+1,cerrent_po[1]]


status_logic = None
def check_tof_wall(tof_now):
    global cerrent_po,status_logic
    if tof_now ==True:

        if robo_status_now =='N':
            grid[cerrent_po[0]][cerrent_po[1]][0] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have west wall             

        if robo_status_now =='E':  
            grid[cerrent_po[0]][cerrent_po[1]][2] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have north wall             

        if robo_status_now =='S':
            grid[cerrent_po[0]][cerrent_po[1]][3] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have east wall   

        if robo_status_now =='W':
            grid[cerrent_po[0]][cerrent_po[1]][1] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have south wall 

def check_left_wall(sharp_l_now):
    global cerrent_po,status_logic
    if sharp_l_now ==True:

        if robo_status_now =='N':
            grid[cerrent_po[0]][cerrent_po[1]][1] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have west wall             

        if robo_status_now =='E':  
            grid[cerrent_po[0]][cerrent_po[1]][0] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have north wall             

        if robo_status_now =='S':
            grid[cerrent_po[0]][cerrent_po[1]][2] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have east wall   

        if robo_status_now =='W':
            grid[cerrent_po[0]][cerrent_po[1]][3] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have south wall  

def check_right_wall(sharp_r_now):
    global cerrent_po,status_logic
    if sharp_r_now == True:
        if robo_status_now =='N':
            grid[cerrent_po[0]][cerrent_po[1]][2] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have east wall             

        if robo_status_now =='E':  
            grid[cerrent_po[0]][cerrent_po[1]][3] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have back wall             

        if robo_status_now =='S':
            grid[cerrent_po[0]][cerrent_po[1]][1] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have west wall   

        if robo_status_now =='W':
            grid[cerrent_po[0]][cerrent_po[1]][0] = 2 #grid[cerrent_po[0]][cerrent_po[1]] have front wall  
    
def check_all_wall (tof,l,r):
    check_tof_wall(tof)
    check_left_wall(l)
    check_right_wall(r)

f_wall = None
l_wall = None
r_wall = None
def check_2_wall():
    global f_wall,l_wall,r_wall
    if robo_status_now =='N':
        if grid[cerrent_po[0]][cerrent_po[1]][0] == 2:
            f_wall = True
        else:
            f_wall = False
        
        if grid[cerrent_po[0]][cerrent_po[1]][1] == 2:
            l_wall = True
        else:
            l_wall = False

        if grid[cerrent_po[0]][cerrent_po[1]][2] == 2:
            r_wall = True
        else:
            r_wall = False

        if grid[cerrent_po[0]][cerrent_po[1]][2] == 2:
            b_wall = True
        else:
            b_wall = False
        
    
    if robo_status_now =='E':
        if grid[cerrent_po[0]][cerrent_po[1]][0] == 2:
            l_wall = True
        
        else:
            l_wall = False

        if grid[cerrent_po[0]][cerrent_po[1]][2] == 2:
            f_wall = True
        
        else:
            f_wall = False
        
        if grid[cerrent_po[0]][cerrent_po[1]][3] == 2:
            r_wall = True
        
        else:
            r_wall = False


    if robo_status_now =='S':
        
        if grid[cerrent_po[0]][cerrent_po[1]][1] == 2:
            r_wall = True
        
        else:
            r_wall = False

        if grid[cerrent_po[0]][cerrent_po[1]][2] == 2:
            l_wall = True   
        
        else:
            l_wall = False

        if grid[cerrent_po[0]][cerrent_po[1]][3] == 2:
            f_wall = True
        
        else:
            f_wall = False

    if robo_status_now =='W':
        if grid[cerrent_po[0]][cerrent_po[1]][0] == 2:
            r_wall = True
        
        else:
            r_wall = False
        
        if grid[cerrent_po[0]][cerrent_po[1]][1] == 2:
            f_wall = True
        
        else:
            f_wall = False

        if grid[cerrent_po[0]][cerrent_po[1]][3] == 2:
            l_wall = True
        
        else:
            l_wall = False

def is_tile_visited(x, y):
    return grid[x][y][4] == 1


''' ----- Logic ----- '''
def logic():
    global status_logic, check 

    row, col = cerrent_po    
    # print("visit_counts =", visit_counts)

    if visit_counts[row][col] < 2:
        # Use TOF sensor for wall checks
        ep_gimbal.moveto(pitch=0, yaw=-90, pitch_speed=300, yaw_speed=300).wait_for_completed()
        time.sleep(0.2)
        sl = tof_distance
        
        ep_gimbal.moveto(pitch=0, yaw= 0, pitch_speed=300, yaw_speed=300).wait_for_completed()
        time.sleep(0.2)
        tof = tof_distance
        
        ep_gimbal.moveto(pitch=0, yaw= 90, pitch_speed=300, yaw_speed=300).wait_for_completed()
        time.sleep(0.2)
        sr = tof_distance

        ep_gimbal.recenter(yaw_speed=400).wait_for_completed()
        # print(f'L{sl} , F{tof} , R{sr}')
        
        if sl < 400:
            status_sl = True
        else:
            status_sl = False
        
        if sr < 400:
            status_sr = True
        
        else:
            status_sr = False
        
        if tof < 400:
            status_of_tof = True
        
        else:
            status_of_tof = False
        
        check_all_wall(status_of_tof,status_sl,status_sr)
        check_2_wall()

    else:
        check_2_wall()


    # print(robo_status_now)
    # print(f'L_wall:{l_wall} , F_wall:{f_wall} , R_wall:{r_wall}')
    
    if f_wall == False and r_wall == True:
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

        detect()
        move_forward()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        adjust_wall_f()
        if adc_r_new < 40:
            adjust_wall_r()
        elif adc_l_new < 40:
            adjust_wall_l()
        status_logic = 'move_forward'

    elif r_wall == False:     
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)       
        turn_right()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        adjust_angle()

        detect()
        move_forward()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        adjust_wall_f()
        if adc_r_new < 40:
            adjust_wall_r()
        elif adc_l_new < 40:
            adjust_wall_l()
        status_logic = 'turn right'

    elif f_wall == True and r_wall == True and l_wall == True:
        turn_back()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        adjust_angle()

        detect()
        move_forward()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        adjust_wall_f()
        if adc_r_new < 40:
            adjust_wall_r()
        elif adc_l_new < 40:
            adjust_wall_l()
        status_logic = 'turn back'

    elif f_wall == True and r_wall == True:
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        turn_left()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0) 
        adjust_angle()

        detect()
        move_forward()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        adjust_wall_f()
        if adc_r_new < 40:
            adjust_wall_r()
        elif adc_l_new < 40:
            adjust_wall_l()
        status_logic = 'turn left'
    
    print('current_position =', cerrent_po)
    print('direction_robot =', robo_status_now)

''' ----- Display map ----- '''
def print_pretty_grid(grid):
    
    # print(top_border)
    
    for row in range(6):
        # Print top walls
        top_line = "|"
        mid_line = "|"
        bot_line = "|"
        for col in range(6):
            cell = grid[row][col]
            north_wall = cell[0]
            west_wall = cell[1]
            east_wall = cell[2]
            south_wall = cell[3]
            
            # Top wall (N)
            top_line += f" {'_' if north_wall == 2 else ' ' }   "
            
            # Middle line with walls (W to E)
            middle = ' V ' if cell[4] == 1 else ' ? '  # Use V for visited cells, ? for unvisited
            mid_line += f"{'|' if west_wall == 2 else ' '}{middle}{'|' if east_wall == 2 else ' '}"
            
            # Bottom wall (S)
            bot_line += f" {'_' if south_wall == 2 else ' ' }   "
        
        print(top_line + "|")
        print(mid_line + "|")
        print(bot_line + "|")
        print("|\t\t\t\t\t|")  # spacer line
    
    # print(bottom_border)

def check_all_cells_visited(grid):
    for row in range(6):
        for col in range(6):
            if grid[row][col][4] == 0:  # If any cell is not visited
                return False
    return True


if __name__ == "__main__":
    ep_robot = robot.Robot()
    print("Initializing robot...")
    ep_robot.initialize(conn_type="ap")

    ep_sensor = ep_robot.sensor
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_camera = ep_robot.camera
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    time.sleep(2)

    center_x = 870 / 2
    center_y = 350 / 2

    p = 0.35
    i = p / (0.7 / 2)
    d = p * (0.7 / 8)
    # p = 0.4705
    # i = 1.1192
    # d = 0.0494

    accumulate_err_x = 0
    accumulate_err_y = 0
    data_pitch_yaw = []
    prev_time = time.time()

    check = True
    found_acrylic = False
    found_chick = False
    count = 0

    ep_chassis.sub_position(freq=10, callback=sub_position_handler)
    ep_sensor.sub_distance(freq=10, callback=tof_data_handler)
    ep_chassis.sub_attitude(freq=10, callback=sub_attitude_info_handler)
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)


    # ปรับตำแหน่ง Gimbal ให้ตรงศูนย์
    ep_gimbal.recenter(pitch_speed=400, yaw_speed=400).wait_for_completed()
    # adjust_wall()
    # adjust_all_walls()
    grid[cerrent_po[0]][cerrent_po[1]][4] = 1

    try:  

        maze_complete = False
        while not maze_complete:
            update_wall()

            if found_chick:
                list_chick.append(tuple(cerrent_po))
                found_chick = False
            if found_acrylic:
                list_acrylic.append(tuple(cerrent_po))
                found_acrylic = False
            if check_all_cells_visited(grid) :
                print("Maze exploration complete! All cells have been visited.")
                maze_complete = True
                ep_chassis.drive_speed(x=0, y=0, z=0)
                ep_robot.close()
            
                # Save the travel data to CSV
                save_travel_data(direction_robot, list_travel)
                save_data_acrylic(list_acrylic)
                save_data_chicken(list_chick)
                break
            print(list_acrylic)
            print(list_chick)
            

    except KeyboardInterrupt:
        print("Program stopped by user. Saving travel data before exit...")
        save_travel_data(direction_robot, list_travel)  # Save data when interrupted
        save_data_acrylic(list_acrylic)
        save_data_chicken(list_chick)
    except Exception as e:
        print(f"An error occurred: {e}")
        save_travel_data(direction_robot, list_travel)  # Save data in case of error
        save_data_acrylic(list_acrylic)
        save_data_chicken(list_chick)
    finally:
        print("Cleaning up...")
        print(grid)
        print_pretty_grid(grid)
        ep_blaster.set_led(brightness=255, effect=blaster.LED_OFF)
        ep_chassis.unsub_position()
        ep_chassis.unsub_attitude()
        ep_sensor.unsub_distance()
        ep_chassis.drive_speed(x=0, y=0, z=0)
        ep_robot.close()
        print("Program ended...")