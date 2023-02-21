from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    stage = LaunchConfiguration('stage', default='1')

    turtlebot3_dqn_pkg_dir = get_package_share_directory('turtlebot3_gazebo')

    return LaunchDescription([
        DeclareLaunchArgument(
            'stage',
            default_value='1',
            description='Stage number'
        ),
        LogInfo(msg=[
            'Stage: [', stage, ']'
        ]),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource([
                turtlebot3_dqn_pkg_dir, '/launch/turtlebot3_dqn_stage', stage, '.launch.py'
            ]),
            launch_arguments={
                'stage': stage
            }.items()
        )
    ])

