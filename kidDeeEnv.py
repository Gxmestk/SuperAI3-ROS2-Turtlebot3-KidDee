#import gym 
import sys
import math
from numpy.core.numeric import Infinity
import numpy
import copy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, Twist
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


from common import utilities as util
from common.settings import ENABLE_BACKWARD, EPISODE_TIMEOUT_SECONDS, ENABLE_MOTOR_NOISE
from common.reward import UNKNOWN, SUCCESS, COLLISION_WALL, TIMEOUT
from common import reward as rw

from turtlebot3_msgs.srv import DrlStep, Goal, RingGoal
#cp file to this directory and drl_gazebo
NUM_SCAN_SAMPLES = util.get_scan_count()
LINEAR = 0
ANGULAR = 1
ENABLE_DYNAMIC_GOALS = False

ACTION_LINEAR_MAX   = 0.22  # in m/s
ACTION_ANGULAR_MAX  = 2.0   # in rad/s

# in meters
ROBOT_MAX_LIDAR_VALUE   = 12
MAX_LIDAR_VALUE         = 4.0

MINIMUM_COLLISION_DISTANCE  = 0.13
MINIMUM_GOAL_DISTANCE       = 0.20


ARENA_LENGTH    = 4
ARENA_WIDTH     = 4
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)


class kidDeeEnv(Node):
    def __init__(self):

        super().__init__('kidDeeEnv')

        """************************************************************
        ** Initialise ROS Topic's Name
        ************************************************************"""

        self.scan_topic = 'scan'
        self.vel_topic = 'cmd_vel'
        self.goal_topic = 'goal_pose'
        self.odom_topic = 'odom'

        """************************************************************
        ** Initialise Robot's Position and Goal's Position
        ************************************************************"""

        self.goal_x, self.goal_y = 0.0, 0.0
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0

        """************************************************************
        ** Timing
        ************************************************************"""


        self.reset_deadline = False
        self.episode_timeout = EPISODE_TIMEOUT_SECONDS
        self.time_sec = 0
        self.clock_msgs_skipped = 0
        self.episode_deadline = Infinity

        """************************************************************
        ** Stepping
        ************************************************************"""

        self.local_step = 0

        """************************************************************
        ** Reset and Done Episode
        ************************************************************"""
        self.succeed = UNKNOWN
        self.done = False
        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = MAX_GOAL_DISTANCE
        self.initial_distance_to_goal = MAX_GOAL_DISTANCE
        self.scan_ranges = [MAX_LIDAR_VALUE] * NUM_SCAN_SAMPLES


        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""

        qos = QoSProfile(depth=10)
        qos_clock = QoSProfile(depth=1)

        # publishers

        self.cmd_vel_pub = self.create_publisher(Twist, self.vel_topic, qos)

        # subscribers

        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)

        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        # clients
        self.task_succeed_client = self.create_client(RingGoal, 'task_succeed')
        self.task_fail_client = self.create_client(RingGoal, 'task_fail')
        # servers
        self.step_comm_server = self.create_service(DrlStep, 'step_comm', self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, 'goal_comm', self.goal_comm_callback)


    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
        # ===================================================================== #
        #                                 Goal                                  #
        # ===================================================================== #

    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True

    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response
        # ===================================================================== #
        #                                 Scan                                  #
        # ===================================================================== #


    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        # noramlize laser values
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = numpy.clip(float(msg.ranges[i]) / MAX_LIDAR_VALUE, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= MAX_LIDAR_VALUE


        # ===================================================================== #
        #                                 Odom                                  #
        # ===================================================================== #

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        _, _, self.robot_heading = util.euler_from_quaternion(msg.pose.pose.orientation)

        # calculate traveled distance for logging
        if self.local_step % 32 == 0:
            self.total_distance += math.sqrt(
                (self.robot_x_prev - self.robot_x)**2 +
                (self.robot_y_prev - self.robot_y)**2)
            self.robot_x_prev = self.robot_x
            self.robot_y_prev = self.robot_y

        diff_y = self.goal_y - self.robot_y
        diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle

        # ===================================================================== #
        #                                Clock                                  #
        # ===================================================================== #

    def clock_callback(self, msg):
        self.time_sec = msg.clock.sec

        if self.reset_deadline:
            self.clock_msgs_skipped += 1

            if self.clock_msgs_skipped > 10: # Wait a few message for simulation to reset clock
                episode_time = self.episode_timeout
                self.episode_deadline = self.time_sec + episode_time
                self.reset_deadline = False
                self.clock_msgs_skipped = 0



        # ===================================================================== #
        #                              New  State                               #
        # ===================================================================== #


    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist()) # stop robot
        self.episode_deadline = Infinity
        self.done = True
        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = numpy.clip(self.difficulty_radius, 0.5, 4)
        if success:
            self.difficulty_radius *= 1.01
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('success service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            self.difficulty_radius *= 0.99
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('fail service not available, waiting again...')
            self.task_fail_client.call_async(req)

        
    def initalize_episode(self, response):
        self.initial_distance_to_goal = self.goal_distance
        response.state = self.get_state(0, 0)
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        rw.reward_initalize(self.initial_distance_to_goal)
        return response


    def get_state(self, action_linear_previous, action_angular_previous):
        state = copy.deepcopy(self.scan_ranges)                                             # range: [ 0, 1]
        state.append(float(numpy.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))     # range: [ 0, 1]
        state.append(float(self.goal_angle) / math.pi)                                      # range: [-1, 1]
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
        self.local_step += 1

        if self.local_step > 30: # Grace period
            # Success
            if self.goal_distance < MINIMUM_GOAL_DISTANCE:
                print("Outcome: Goal reached! :)")
                self.succeed = SUCCESS
            # Collision
            elif self.obstacle_distance < MINIMUM_COLLISION_DISTANCE:
                
                

                    print("Outcome: Collision! (wall) :(")
                    self.succeed = COLLISION_WALL
            # Timeout
            elif self.time_sec >= self.episode_deadline:
                print("Outcome: Time out! :(")
                self.succeed = TIMEOUT
            # Tumble
            elif self.robot_tilt > 0.06 or self.robot_tilt < -0.06:
                print("Outcome: Tumble! :(")
                self.succeed = TUMBLE
            if self.succeed is not UNKNOWN:
                self.stop_reset_robot(self.succeed == SUCCESS)
        return state


    def step_comm_callback(self, request, response):
        if len(request.action) == 0:
            return self.initalize_episode(response)

        if ENABLE_MOTOR_NOISE:
            request.action[LINEAR] += numpy.clip(numpy.random.normal(0, 0.05), -0.1, 0.1)
            request.action[ANGULAR] += numpy.clip(numpy.random.normal(0, 0.05), -0.1, 0.1)

        # Un-normalize actions
        if ENABLE_BACKWARD:
            action_linear = request.action[LINEAR] * ACTION_LINEAR_MAX
        else:
            action_linear = (request.action[LINEAR] + 1) / 2 * ACTION_LINEAR_MAX
        action_angular = request.action[ANGULAR]*ACTION_ANGULAR_MAX

        # Publish action cmd
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        # Prepare repsonse
        response.state = self.get_state(request.previous_action[LINEAR], request.previous_action[ANGULAR])
        response.reward = rw.get_reward(self.succeed, action_linear, action_angular, self.goal_distance,
                                            self.goal_angle, self.obstacle_distance)
        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0
        if self.done:
            response.distance_traveled = self.total_distance
            # Reset variables
            self.succeed = UNKNOWN
            self.total_distance = 0.0
            self.local_step = 0
            self.done = False
            self.reset_deadline = True
        if self.local_step % 200 == 0:
            print(f"Rtot: {response.reward:.3f}, GD: {self.goal_distance:.3f}, GA: {math.degrees(self.goal_angle):.3f}?? \
                    MinD: {self.obstacle_distance:.3f}, Alin: {action_linear:.3f}, Aturn: {action_angular:.3f}")
        return response

        # ===================================================================== #
        #                                 Main                                  #
        # ===================================================================== #


def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    if len(args) == 0:
        kidDeeEnv_ = kidDeeEnv()
    else:
        rclpy.shutdown()
        quit("ERROR: wrong number of arguments!")
    rclpy.spin(kidDeeEnv_)
    kidDeeEnv.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()