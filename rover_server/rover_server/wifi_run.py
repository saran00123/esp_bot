import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from scipy.interpolate import interp1d
import socket
import json

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        self.HOST = '192.168.1.17'
        self.PORT = 80
        self.speed = [0, 0, 0, 0]

    def value_map(self, value):
        if value < 5.0:
            value = 5.0
        elif value > 45.0:
            value = 45.0
        print(value)
        self.inter = interp1d([5, 45],[0,255])
        return self.inter(value)


    def cmd_vel_callback(self, msg):
        # Constants for robot parameters (to be adjusted according to your robot)
        wheel_separation = 0.3  # Distance between the wheels in meters
        wheel_radius = 0.035  # Radius of the wheels in meters

        # Extract linear and angular velocity components
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Calculate left and right wheel speeds
        left_speed = (linear_x - angular_z * (wheel_separation / 2)) / wheel_radius
        right_speed = (linear_x + angular_z * (wheel_separation / 2)) / wheel_radius
        # print("left speed Fwd : "+ str(left_speed))
        # print("right speed Fwd : "+ str(right_speed))
        if (left_speed < 0):
            left_speed_f = 5.0
            left_speed_r = left_speed * (-1.0)
        else:
            left_speed_r = 5.0
            left_speed_f = left_speed

        if (right_speed < 0):
            right_speed_f = 5.0
            right_speed_r = right_speed * (-1.0)
        else:
            right_speed_r = 5.0
            right_speed_f = right_speed
        
        self.speed[0] = int(self.value_map(right_speed_f))
        self.speed[1] = int(self.value_map(right_speed_r))
        self.speed[2] = int(self.value_map(left_speed_f))
        self.speed[3] = int(self.value_map(left_speed_r))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            message = json.dumps(self.speed)
            s.sendall(message.encode())
            print("Sent:", message)
            
        print("left speed Fwd : "+ str(self.speed[2]))
        print("right speed Fwd : "+ str(self.speed[0]))
        print("left speed Rev : "+ str(self.speed[3]))
        print("right speed Rev : "+ str(self.speed[1]))

def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    rclpy.spin(motor_controller)
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
