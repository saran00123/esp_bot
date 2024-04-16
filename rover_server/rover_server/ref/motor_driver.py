#!/usr/bin/env python2
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from motor_drivers.msg import left_encoder,right_encoder
import time,math
import serial,json
from datetime import datetime
global busy,speed_write_status,speed_write_right_status,busy
busy = False
speed_write_status = False
speed_write_right_status = False
busy = False
last_x_linear = 0.0
last_z_linear = 0.0
def read_write(msg):
    global busy
    try:
        msg = json.dumps(msg) + '\r\n'
        busy = True
        print("Write    -- ",msg)
        ser.write(msg)
        read_message = ser.readline()
        print("Read  ---- ",read_message)
        busy = False
        return read_message
    except Exception as error:
        print("Error in read write {}".format(error))

def write(msg):
    global busy
    try:
        msg = json.dumps(msg) + '\r\n'
        busy = True
        ser.write(msg)
        time.sleep(0.01)
        busy = False
    except Exception as error:
        print("Error in read write {}".format(error))
        
def encoder_left_reset_cb():
    global busy,speed_write_status
    try:
        outgoing_json = {}
        outgoing_json["reset_left_encoder"] = True
        if busy == True:
            speed_write_status = True
            time.sleep(0.2)
        get_encoder_reset_message = read_write(outgoing_json)
        print(get_encoder_reset_message)
        speed_write_status = False
        if len(get_encoder_reset_message) > 0 :
            incoming_message = json.loads(get_encoder_reset_message)
            # return LeftEncoderResetResponse(response = incoming_message["reset_left_encoder"])
    except Exception as error:
        print("Error in the encoder reset callback {}".format(error))

def encoder_right_reset_cb():
    global busy,speed_write_status
    try:
        outgoing_json = {}
        outgoing_json["reset_right_encoder"] = True
        if busy == True:
            speed_write_status = True
            time.sleep(0.2)
        speed_write_status = True
        get_encoder_reset_message = read_write(outgoing_json)
        print(get_encoder_reset_message)
        speed_write_status = False
        if len(get_encoder_reset_message) > 0 :
            incoming_message = json.loads(get_encoder_reset_message)
            # return RightEncoderResetResponse(response = incoming_message["reset_right_encoder"])
    except Exception as error:
        print("Error in the encoder right reset call back {}".format(error))

def get_encoder_value(event):
    global busy,speed_write_status
    try:
        print(speed_write_status)
        if speed_write_status == False:
            #print("Inside the get encoder value")
            encoder_value_json = {}
            encoder_value_json["get_left_encoder"] = True
            if busy == True:
                time.sleep(0.2)
            busy = True
            get_encoder_message = read_write(encoder_value_json)
            # print(get_encoder_message)
            busy = False
            if get_encoder_message:
                left_encoder_value = json.loads(get_encoder_message)
                left_encoder_value = -1 * left_encoder_value["left_encoder"]
                # print(left_encoder_value)
                #left_encoder_pub.value = left_encoder_value
                data = Int32()
                data.data = left_encoder_value
                left_encoder_publisher.publish(data)
    except Exception as error:
        print("Error in the get encoder value {}".format(error))

def get_encoder_right_value(event):
    global busy,speed_write_status
    try:
        if speed_write_status == False:
            encoder_value_json = {}
            encoder_value_json["get_right_encoder"] = True
            if busy == True:
                time.sleep(0.2)
            busy = True
            get_encoder_message = read_write(encoder_value_json)
            busy = False
            if get_encoder_message:
                right_encoder_value = json.loads(get_encoder_message)
                right_encoder_value =  right_encoder_value["right_encoder"]
                print(right_encoder_value)
                data = Int32()
                data.data = right_encoder_value
                # right_encoder_pub.value = right_encoder_value
                right_encoder_publisher.publish(data)
    except Exception as error:
        pass

def motor_speed_cb(msg):
    global busy,speed_write_status,speed_write_right_status,busy,last_x_linear,last_z_linear,motor_name
    try:
        print("Inside the motor speed callback",msg)
        if msg.linear.x != last_x_linear or msg.angular.z != last_z_linear:
            last_x_linear = msg.linear.x
            last_z_linear = msg.angular.z
            robot_width = 0.520
            wheel_radius = 0.095
            gear_ratio = 23
            if motor_name == "left":
                vel = (msg.linear.x - msg.angular.z * robot_width/2.0) / wheel_radius
            else:
                vel = (msg.linear.x + msg.angular.z * robot_width/2.0) / wheel_radius
            c = math.pi * 0.1905
            # print(left_wheel_vel,right_wheel_vel)
            rpm = int((vel)*9.54929659642538) 
            #print("Left rpm : " + str(left_rpm) + " Right rpm : " + str(right_rpm))
            outgoing_json = {}
            outgoing_json_right = {}
            print(motor_name)
            if motor_name == "left":
                outgoing_json["left_rpm"] = rpm
            elif motor_name == "right":
                outgoing_json["right_rpm"] = rpm
            print(outgoing_json)
            if busy == True:
                speed_write_status = True
                time.sleep(0.1)
            speed_write_status = True
            print(datetime.now())
            get_message = read_write(outgoing_json)
            print(get_message)
            speed_write_status = False
    except Exception as error:
        print("Error in the motor speed value {}".format(error))

import sys

if __name__ == "__main__":
    motor_name=sys.argv[1]
    port = sys.argv[2]
    ser = serial.Serial(port = port,baudrate = 115200, timeout = 0.4)
    rospy.init_node(motor_name+"_motor_driver_node")
    if motor_name == "left":
        encoder_left_reset_cb()
        left_encoder_pub = left_encoder()
        # print("Inside")
        left_encoder_publisher = rospy.Publisher("/left_encoder",Int32,queue_size=1)
        print("Inside")
        print(speed_write_status)
        rospy.Timer(rospy.Duration(0.01),get_encoder_value)
    elif motor_name == "right":
        encoder_right_reset_cb()
        right_encoder_pub = right_encoder()
        right_encoder_publisher = rospy.Publisher("/right_encoder",Int32,queue_size=1)
        rospy.Timer(rospy.Duration(0.01),get_encoder_right_value)
        # rospy.Service("/encoder_reset_right",RightEncoderReset,encoder_right_reset_cb)
    rospy.Subscriber("/cmd_vel",Twist,motor_speed_cb)
    rospy.spin()
