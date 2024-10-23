import robomaster
from robomaster import robot
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt

adc_l = None
adc_r = None
MAX_SPEED = 1
WALL_DISTANCE_THRESHOLD = 15.0
count = 0

# ตัวแปรเก็บข้อมูลระยะทางจากเซ็นเซอร์ ToF และสถานะการตรวจจับวัตถุ
tof_distance = 0
status_tof = False
# ตัวแปรเก็บข้อมูลระยะทางล่าสุดที่คำนวณได้จากเซ็นเซอร์ซ้ายและขวา (Sharp Sensors)
adc_r_new = 0
adc_l_new = 0
# ลิสต์สำหรับเก็บข้อมูลระยะทางจากเซ็นเซอร์ซ้ายและขวาในแต่ละช่วงเวลา
left_data = []
right_data = []
left_time_data = []
right_time_data = []
position_data = []
pitch = 0

def sub_position_handler(position_info):
    x, y, z = position_info
    # print("chassis position: x:{:.2f}, y:{:.2f}, z:{:.2f}".format(x, y, z))
    position_data.append((round(x,2), (round(y,2)), (round(z,2))))

def sub_attitude_info_handler(attitude_info):
    global yaw
    yaw, pitch, roll = attitude_info
    print("chassis attitude: yaw:{0}, pitch:{1}, roll:{2} ".format(yaw, pitch, roll))

def tof_data_handler(sub_info):
    global tof_distance, status_tof
    tof_distance = sub_info[0]
    if 100 < tof_distance < 300:
        status_tof = True
    else:
        status_tof = False
    print(f"ToF distance: {tof_distance} mm")

    global adc_r, adc_l, adc_r_new, adc_l_new, status_ss_r, status_ss_l
    adc_r = ep_sensor_adaptor.get_adc(id=2, port=2)
    adc_r_cm = (adc_r * 3) / 1023  # process to cm unit
    adc_l = ep_sensor_adaptor.get_adc(id=1, port=1)
    adc_l_cm = (adc_l * 3) / 1023  # process to cm unit

    if adc_r_cm > 1.4:
        adc_r_new = ((adc_r_cm - 4.2) / -0.31) - 6
    elif 1.4 >= adc_r_cm >= 0.6:
        adc_r_new = ((adc_r_cm - 2.03) / -0.07) - 6
    elif 0 <= adc_r_cm < 0.6:
        adc_r_new = ((adc_r_cm - 0.95) / -0.016) - 6

    if adc_l_cm > 1.4:
        adc_l_new = ((adc_l_cm - 4.2) / -0.31) - 6
    elif 1.4 >= adc_l_cm >= 0.6:
        adc_l_new = ((adc_l_cm - 2.03) / -0.07) - 6
    elif 0 <= adc_l_cm < 0.6:
        adc_l_new = ((adc_l_cm - 0.95) / -0.016) - 6

    # print(f"distance from front wall:right  {adc_r} left  {adc_l}")
    print(f"distance from front wall:right  {adc_r_new} left  {adc_l_new}")

    if 30 > adc_r_new > 2:
        status_ss_r = True
    else:
        status_ss_r = False

    if 30 > adc_l_new > 2:
        status_ss_l = True
    else:
        status_ss_l = False

if __name__ == "__main__":
    # เริ่มต้นการทำงานของหุ่นยนต์
    ep_robot = robot.Robot()
    print("Initializing robot...")
    ep_robot.initialize(conn_type="ap")
    time.sleep(2)  # รอ 2 วินาทีหลังการเชื่อมต่อ

    # เริ่มต้นการทำงานของเซ็นเซอร์ต่าง ๆ
    ep_sensor = ep_robot.sensor
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor

    # สมัครสมาชิกฟังก์ชัน callback เพื่อรับข้อมูลจากเซ็นเซอร์ ToF และ Sharp Sensors
    ep_chassis.sub_position(freq=10, callback=sub_position_handler)
    ep_sensor.sub_distance(freq=10, callback=tof_data_handler)
    ep_chassis.sub_attitude(freq=10, callback=sub_attitude_info_handler)

    # ep_sensor_adaptor.sub_adapter(freq=5, callback=sub_data_handler)

    # ปรับตำแหน่ง Gimbal ให้ตรงศูนย์
    ep_gimbal.recenter().wait_for_completed()
    try:
        plt.ion()
        fig, ax = plt.subplots()

        while True:
            count += 1 
            print("current count = {}".format(count))

            if tof_distance is None or adc_l is None or adc_r is None:
                print("Waiting for sensor data...")
                time.sleep(1)
                continue

            gap = abs(WALL_DISTANCE_THRESHOLD - adc_r_new)
            walk_y = 0.055
            
            while True:
                if position_data:
                    x_vals = [pos[0] for pos in position_data]
                    y_vals = [pos[1] for pos in position_data]
                    ax.clear()
                    ax.plot(y_vals, x_vals, '*-', label="Robot Path")
                    ax.set_xlabel('Y Position (cm)')
                    ax.set_ylabel('X Position (cm)')
                    ax.set_title('Real-time Robot Path')
                    ax.grid(True)
                    plt.draw()
                    plt.pause(0.01)
    except:
        pass  # Added exception handling