# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from tf2_ros import TransformBroadcaster
from go2_interfaces.msg import Go2State
from std_msgs.msg import Header

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Imu
from sensor_msgs_py import point_cloud2

from isaaclab.sensors import CameraCfg, Camera
import omni.kit.commands
import omni.replicator.core as rep
import isaaclab.sim as sim_utils

from pxr import UsdGeom, UsdPhysics, Gf, Sdf, UsdPhysics
from tf_transformations import quaternion_from_euler

UnitreeL1_translation = (0.293, 0.0, -0.08)
UnitreeL1_quat = quaternion_from_euler(0, 165 * 3.14159265 / 180, 0) # 165° y axis

ExtraLidar_translation = (0.15, 0.0, 0.18)
ExtraLidar_quat = quaternion_from_euler(0, 0, 0) 

def add_rtx_lidar(num_envs, robot_type, lidar_type, debug=False):
    if lidar_type == "UnitreeL1":
        trans = UnitreeL1_translation
        quat = UnitreeL1_quat
        config = "Unitree_L1"
    if lidar_type == "Extra":
        trans = ExtraLidar_translation
        quat = ExtraLidar_quat
        config = "Unitree_L1_old"
    
    annotator_lst = []
    for i in range(num_envs):

        _, lidar_sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=f"/World/envs/env_{i}/Robot/base/lidar_sensor",
            parent=None,
            translation=trans,
            orientation=Gf.Quatd(quat[3], quat[0], quat[1], quat[2]),
            config=config,
        )
        # if lidar_type == "Extra":
        #     attach_usd_to_sensor(lidar_sensor.GetPath(), "./lidar/os2_mesh.usd")

        lidar_texture = rep.create.render_product(lidar_sensor.GetPath(), [1, 1], name="UnitreeL1")
        if debug:
            # Create the debug draw pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloudBuffer")
            writer.attach([lidar_texture])

        # writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        # writer.initialize(topicName=f"robot{i}/point_cloud2", frameId=f"robot{i}/base_link")
        # writer.attach([lidar_texture])

        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        annotator.attach(lidar_texture)

        annotator_info = {
            "type": lidar_type,
            "annotator_object": annotator
        }

        annotator_lst.append(annotator_info)
    
    return annotator_lst

def attach_usd_to_sensor(sensor_path: str, usd_path: str, visible=True):
    stage = omni.usd.get_context().get_stage()
    visual_path = f"{sensor_path}/visual"
    UsdGeom.Xform.Define(stage, Sdf.Path(visual_path))
    mesh_prim = stage.DefinePrim(Sdf.Path(f"{visual_path}/mesh"), "Xform")
    mesh_prim.GetReferences().AddReference(usd_path)
    
    UsdPhysics.CollisionAPI.Apply(mesh_prim).CreateCollisionEnabledAttr(False)
    
    # rt_vis_api = UsdGeom.Tokens
    mesh_prim.CreateAttribute("visibility:raytracing:camera", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:transmission", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:shadow", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:diffuse", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:glossy", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:specular", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:scatter", Sdf.ValueTypeNames.Token).Set("invisible")


    if visible:
        path = Sdf.Path(sensor_path)
        prim = stage.GetPrimAtPath(path)
        prim.GetAttribute("visibility").Set("inherited")

def add_camera(num_envs):
    for i in range(num_envs):
        cameraCfg = CameraCfg(
            prim_path=f"/World/envs/env_{i}/Robot/base/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.32487, -0.00095, 0.05362),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

        Camera(cameraCfg)

def add_copter_camera():

    cameraCfg = CameraCfg(
        prim_path=f"/World/Crazyflie/Robot_1/body/front_cam_2",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.2), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"
        ),
    )

    Camera(cameraCfg)

def pub_robo_data_ros2(num_envs, base_node, env, annotator_lst):

    for i in range(num_envs):
        # publish ros2 info
        base_node.publish_joints(
            env.unwrapped.scene["robot"].data.joint_names,
            env.unwrapped.scene["robot"].data.joint_pos[i],
            i,
        )
        base_node.publish_odom(
            env.unwrapped.scene["robot"].data.root_state_w[i, :3],
            env.unwrapped.scene["robot"].data.root_state_w[i, 3:7],
            i,
        )
        base_node.publish_imu(
            env.unwrapped.scene["robot"].data.root_state_w[i, 3:7],
            env.unwrapped.scene["robot"].data.root_lin_vel_b[i, :],
            env.unwrapped.scene["robot"].data.root_ang_vel_b[i, :],
            i,
        )
        
        base_node.publish_robot_state(
            [
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][4][2],
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][8][2],
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][14][2],
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][18][2],
            ],
            i,
        )

        try:
            for lidar_id in range(2): 
                index_annotator = (i * 2) + lidar_id
                base_node.publish_lidar(annotator_lst[index_annotator], i)

        except Exception as e:
            print(f"Erro ao publicar LiDAR para o ambiente {i}: {e}")


class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__("go2_driver_node")
        qos_profile = QoSProfile(depth=10)

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_L1_pub = []
        self.go2_lidar_extra_pub = []
        self.odom_pub = []
        self.imu_pub = []

        for i in range(num_envs):
            self.joint_pub.append(
                self.create_publisher(JointState, f"robot{i}/joint_states", qos_profile)
            )
            self.go2_state_pub.append(
                self.create_publisher(Go2State, f"robot{i}/go2_states", qos_profile)
            )
            self.odom_pub.append(
                self.create_publisher(Odometry, f"robot{i}/odom", qos_profile)
            )
            self.imu_pub.append(
                self.create_publisher(Imu, f"robot{i}/imu", qos_profile)
            )
            self.go2_lidar_L1_pub.append(
                self.create_publisher(
                    PointCloud2, f"robot{i}/point_cloud2_L1", qos_profile
                )
            )
            self.go2_lidar_extra_pub.append(
                self.create_publisher(
                    PointCloud2, f"robot{i}/point_cloud2_extra", qos_profile
                )
            )
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

    def publish_joints(self, joint_names_lst, joint_state_lst, robot_num):
        # Create message
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()

        joint_state_names_formated = []
        for joint_name in joint_names_lst:
            joint_state_names_formated.append(f"robot{robot_num}/" + joint_name)

        joint_state_formated = []
        for joint_state_val in joint_state_lst:
            joint_state_formated.append(joint_state_val.item())

        joint_state.name = joint_state_names_formated
        joint_state.position = joint_state_formated
        self.joint_pub[robot_num].publish(joint_state)

    def publish_odom(self, base_pos, base_rot, robot_num):
        now = self.get_clock().now().to_msg()

        odom_trans = TransformStamped()
        odom_trans.header.stamp = now
        odom_trans.header.frame_id = "odom"
        odom_trans.child_frame_id = f"robot{robot_num}/base_link"
        odom_trans.transform.translation.x = base_pos[0].item()
        odom_trans.transform.translation.y = base_pos[1].item()
        odom_trans.transform.translation.z = base_pos[2].item()
        odom_trans.transform.rotation.x = base_rot[1].item()
        odom_trans.transform.rotation.y = base_rot[2].item()
        odom_trans.transform.rotation.z = base_rot[3].item()
        odom_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(odom_trans)

        UnitreeL1_trans = TransformStamped()
        UnitreeL1_trans.header.stamp = now
        UnitreeL1_trans.header.frame_id = f"robot{robot_num}/base_link"
        UnitreeL1_trans.child_frame_id = f"robot{robot_num}/UnitreeL1_link"
        UnitreeL1_trans.transform.translation = Vector3(x=UnitreeL1_translation[0], y=UnitreeL1_translation[1], z=UnitreeL1_translation[2])
        UnitreeL1_trans.transform.rotation = Quaternion(x=UnitreeL1_quat[0], y=UnitreeL1_quat[1], z=UnitreeL1_quat[2], w=UnitreeL1_quat[3])
        self.broadcaster.sendTransform(UnitreeL1_trans)

        lidar_trans = TransformStamped()
        lidar_trans.header.stamp = now
        lidar_trans.header.frame_id = f"robot{robot_num}/base_link"
        lidar_trans.child_frame_id = f"robot{robot_num}/lidar_link"
        lidar_trans.transform.translation = Vector3(x=ExtraLidar_translation[0], y=ExtraLidar_translation[1], z=ExtraLidar_translation[2])
        lidar_trans.transform.rotation = Quaternion(x=ExtraLidar_quat[0], y=ExtraLidar_quat[1], z=ExtraLidar_quat[2], w=ExtraLidar_quat[3])
        self.broadcaster.sendTransform(lidar_trans)

        odom_topic = Odometry()
        odom_topic.header.stamp = now
        odom_topic.header.frame_id = "odom"
        odom_topic.child_frame_id = f"robot{robot_num}/base_link"
        odom_topic.pose.pose.position.x = base_pos[0].item()
        odom_topic.pose.pose.position.y = base_pos[1].item()
        odom_topic.pose.pose.position.z = base_pos[2].item()
        odom_topic.pose.pose.orientation.x = base_rot[1].item()
        odom_topic.pose.pose.orientation.y = base_rot[2].item()
        odom_topic.pose.pose.orientation.z = base_rot[3].item()
        odom_topic.pose.pose.orientation.w = base_rot[0].item()
        self.odom_pub[robot_num].publish(odom_topic)
        
    def publish_imu(self, base_rot, base_lin_vel, base_ang_vel, robot_num):
        imu_trans = Imu()
        imu_trans.header.stamp = self.get_clock().now().to_msg()
        imu_trans.header.frame_id = f"robot{robot_num}/base_link"

        imu_trans.linear_acceleration.x = base_lin_vel[0].item()
        imu_trans.linear_acceleration.y = base_lin_vel[1].item()
        imu_trans.linear_acceleration.z = base_lin_vel[2].item()

        imu_trans.angular_velocity.x = base_ang_vel[0].item()
        imu_trans.angular_velocity.y = base_ang_vel[1].item()
        imu_trans.angular_velocity.z = base_ang_vel[2].item()

        imu_trans.orientation.x = base_rot[1].item()
        imu_trans.orientation.y = base_rot[2].item()
        imu_trans.orientation.z = base_rot[3].item()
        imu_trans.orientation.w = base_rot[0].item()

        self.imu_pub[robot_num].publish(imu_trans)

    def publish_robot_state(self, foot_force_lst, robot_num):

        go2_state = Go2State()
        go2_state.foot_force = [
            int(foot_force_lst[0].item()),
            int(foot_force_lst[1].item()),
            int(foot_force_lst[2].item()),
            int(foot_force_lst[3].item()),
        ]
        self.go2_state_pub[robot_num].publish(go2_state)

    def publish_lidar(self, data, robot_num):
        points = data["annotator_object"].get_data()["data"]

        point_cloud = PointCloud2()
        if data["type"] == "UnitreeL1":
            point_cloud.header = Header(frame_id=f"robot{robot_num}/UnitreeL1_link")
            publisher = self.go2_lidar_L1_pub[robot_num]
        elif data["type"] == "Extra":
            point_cloud.header = Header(frame_id=f"robot{robot_num}/lidar_link")
            publisher = self.go2_lidar_extra_pub[robot_num]

        point_cloud.header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, points)
        publisher.publish(point_cloud)

    async def run(self):
        while True:
            self.publish_lidar()
            await asyncio.sleep(0.1)
