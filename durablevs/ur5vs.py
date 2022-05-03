import time
import numpy as np
import math
import pybullet as p
import pybullet_data as pd
import math
import time
import cv2
import matplotlib.pyplot as plt
import os

class UR5VS:
    def __init__(self):
        self.p = p
        self.start_pybullet()
        self.xarm = self.load_xarm()
        self.aruco_ids = self.load_aruco_markers()

        self.aruco_relative_poses = [
            ( [0.0, 1.0, -1.0], p.getQuaternionFromEuler([0.0, 0.0, np.pi])),
            ([0.9, 0.7, -1.0], p.getQuaternionFromEuler([0.0, 0.0, np.pi-np.pi/4])),
            ([-0.9, 0.7, -1.0], p.getQuaternionFromEuler([0.0, 0.0, np.pi+np.pi/4]))
        ]
        self.home_jps = np.array([0.0, -1.1, 1.9, np.pi + np.pi/4, -np.pi/2, np.pi/2])
        self.current_jps = self.home_jps
        self.go_to_position(self.current_jps)

        self.set_up_cameras()

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

    def get_observations(self):
        images = self.get_images()
        detections = self.run_aruco_detector_on_images(images)
        num_cameras =len(images)
        num_markers = 3
        observations = np.ones((num_cameras, num_markers * 4, 2)) * -1000

        for i in range(num_cameras):
            detection = detections[i]
            if len(detection[0]) == 0:
                continue
            for (det, aruco_id) in zip(detection[0], detection[1]):
                actual_id = aruco_id[0]
                observations[i,4*actual_id : (4*actual_id + 4),:] = det[0,:,:]
        return observations, images, detections

    def get_detection_overlay_images(self, images, observations):
        colors = ["red","blue","green"]
        figures = []
        for (i, img) in enumerate(images):
            fig = plt.figure()
            plt.imshow(img)
            for j in range(3):
                pixels = observations[i,4*j : (4*j + 4),:]
                if np.any(pixels != -1000.0):
                    plt.scatter(pixels[:,0],pixels[:,1], label=str(j), color=colors[j], alpha=0.5)
            plt.legend()

    def get_images(self):
        images = []
        for v in self.camera_view_matrices:
            _,_,rgb_image,depth_image,_ = p.getCameraImage(self.WIDTH, self.HEIGHT, v, self.camera_proj_matrix)
            img = np.array(rgb_image).reshape(self.HEIGHT, self.WIDTH, 4)[:,:,0:3]
            img = img.astype(np.uint8)
            images.append(img)
        return images


    def run_aruco_detector_on_images(self, image_list):
        return [
            cv2.aruco.detectMarkers(img, self.arucoDict, parameters=self.arucoParams)[0:2]
            for img in image_list
        ]

    def get_position(self):
        start = 1
        jps = np.zeros(6)
        for i in range(start,7):
            jps[i-start] =  p.getJointState(self.xarm, i)[0]
        return jps

    def go_to_position(self, jps):
        start = 1
        for i in range(start,7):
        #     print(p.getJointInfo(xarm, i)[2])
            p.changeDynamics(self.xarm, i, linearDamping=0, angularDamping=0)
            p.resetJointState(self.xarm, i, jps[i-start])

        EE_link = 8
        linkstate = p.getLinkState(self.xarm, EE_link)
        pos = linkstate[0]
        orn = linkstate[1]

        for (aruco_id, rel_pos) in zip(self.aruco_ids, self.aruco_relative_poses):
            mpos, morn = p.multiplyTransforms(pos, orn, rel_pos[0], rel_pos[1])
            p.resetBasePositionAndOrientation(aruco_id, mpos, morn)

    # SETUP

    def start_pybullet(self):
        p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        p.setAdditionalSearchPath(pd.getDataPath())

    def load_xarm(self):
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        xarm = p.loadURDF("/home/nishadg/vicarious/robust_visual_servo/learning_visual_control/src/learning_visual_control/urdf/ur5_robotiq_140.urdf",
            np.array([-0,-0,-22]), [0,0,0,1], useFixedBase=True, flags=flags, globalScaling=30.0)
        return xarm

    def load_aruco_markers(self):
        aruco_ids = []
        dirname = os.path.dirname(__file__)
        for i in [0,1,2]:
            aruco_cube_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=os.path.join(dirname, "aruco_markers/marker_" + str(i) + "/textured.obj"))
            aruco_cube_collision_id = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=os.path.join(dirname, "aruco_markers/marker_" + str(i) + "/textured.obj"))

            aruco_id = p.createMultiBody(
                baseVisualShapeIndex=aruco_cube_shape_id,
                baseCollisionShapeIndex=aruco_cube_collision_id,
                basePosition=np.array([0,0,0]),
                baseOrientation=[0,0,0,1],
            )
            aruco_ids.append(aruco_id)
        return aruco_ids

    def set_up_cameras(self):
        eye = np.array([50.0, 0.0, 0.0])
        origin = np.array([0.0, 0.0, 0.0])
        up_vec = np.array([0.0, 0.0, 1.0])

        viewMatrix1 = p.computeViewMatrix(
            cameraEyePosition= np.array([50.0, 0.0, 0.0]),
            cameraTargetPosition=origin,
            cameraUpVector=up_vec,
        )

        viewMatrix2 = p.computeViewMatrix(
            cameraEyePosition= np.array([45.0, 10.0, 10.0]),
            cameraTargetPosition=origin,
            cameraUpVector=up_vec,
        )

        fov = 30.0
        aspect_ratio = 1.5
        near = 0.01
        far = 100.0
        projMatrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)

        self.WIDTH, self.HEIGHT = 1200, 800
        self.cx, self.cy = self.WIDTH / 2.0, self.HEIGHT / 2.0
        fov_y = np.deg2rad(fov)
        fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
        # Use the following relation to recover the focal length:
        #   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
        self.fx = self.cx / np.tan(fov_x / 2.0)
        self.fy = self.cy / np.tan(fov_y / 2.0)

        self.camera_view_matrices = [viewMatrix1, viewMatrix2]
        self.camera_proj_matrix = projMatrix


    #########
