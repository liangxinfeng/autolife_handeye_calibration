from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    prefix = '/control_center_joint_control/'
    names = ['joint_states', 'target_joint_states', 'status', 'preview_command',
             'apply_command', 'action_command', 'set_hardware_enabled',
             'set_hardware_enabled_without_reset', 'set_leg_waist_enabled']
    return LaunchDescription([
        SetEnvironmentVariable('ROS_DOMAIN_ID', '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        SetEnvironmentVariable('CYCLONEDDS_URI', '<CycloneDDS><Domain><General><Interfaces>'
            '<NetworkInterface name="lo"/></Interfaces></General><Discovery>'
            '<ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex>'
            '</Discovery></Domain></CycloneDDS>'),
        Node(package='autolife_handeye_calibration_307', executable='calibration_app',
             parameters=[{'robot_id':307, 'goal_tolerance_deg':0.3, 'gripper_goal_tolerance_deg':1.0,
                          'command_update_rate_hz':10.0, 'max_single_step_deg':5.0,
                          'max_gripper_command_lead_deg':30.0}],
             output='screen', remappings=[(prefix+n, '/handeye307/'+n) for n in names]),
    ])
