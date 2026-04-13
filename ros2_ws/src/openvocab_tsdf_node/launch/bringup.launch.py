"""Launch the openvocab_tsdf grounding node.

    ros2 launch openvocab_tsdf_node bringup.launch.py \\
        map_path:=$HOME/openvocab-tsdf/outputs/demo_map.npz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_path = LaunchConfiguration("map_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_path",
                description="path to a precomputed feature map npz",
            ),
            DeclareLaunchArgument("model", default_value="ViT-B-16"),
            DeclareLaunchArgument("pretrained", default_value="laion2b_s34b_b88k"),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("dtype", default_value="fp16"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            Node(
                package="openvocab_tsdf_node",
                executable="grounding_node",
                name="openvocab_grounding",
                output="screen",
                parameters=[
                    {
                        "map_path": map_path,
                        "model": LaunchConfiguration("model"),
                        "pretrained": LaunchConfiguration("pretrained"),
                        "device": LaunchConfiguration("device"),
                        "dtype": LaunchConfiguration("dtype"),
                        "map_frame": LaunchConfiguration("map_frame"),
                    }
                ],
            ),
        ]
    )
