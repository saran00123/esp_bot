import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        self.left_speed_pub = self.create_publisher(Float32, 'left_motor_speed', 10)
        self.right_speed_pub = self.create_publisher(Float32, 'right_motor_speed', 10)

    def cmd_vel_callback(self, msg):
        # Constants for robot parameters (to be adjusted according to your robot)
        wheel_separation = 0.3  # Distance between the wheels in meters
        wheel_radius = 0.05  # Radius of the wheels in meters

        # Extract linear and angular velocity components
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Calculate left and right wheel speeds
        left_speed = (linear_x - angular_z * (wheel_separation / 2)) / wheel_radius
        right_speed = (linear_x + angular_z * (wheel_separation / 2)) / wheel_radius

        # Publish left and right wheel speeds
        left_msg = Float32()
        left_msg.data = left_speed
        right_msg = Float32()
        right_msg.data = right_speed
        self.left_speed_pub.publish(left_msg)
        self.right_speed_pub.publish(right_msg)

def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    rclpy.spin(motor_controller)
    motor_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
