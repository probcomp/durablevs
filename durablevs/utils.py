from __future__ import print_function

import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from builtins import range

import torch


def generate_aruco_detection_visualization(submat, images, current=None, prefix=""):
    num_cameras, num_points, _ = submat.shape
    assert num_points % 4 == 0
    num_markers = num_points / 4
    color = ['rx', 'gx', 'bx', 'cx']
    color_pred = ['m+', 'y+', 'k+', 'w+']

    for cam_idx in range(num_cameras):
        for idx in range(num_markers):
            plt.clf()
            if np.all(submat[cam_idx, 4 * idx : 4 * idx + 4, :] == -1000) and np.all(
                current[cam_idx, 4 * idx : 4 * idx + 4, :] == -1000
            ):
                continue

            plt.figure()
            plt.imshow(images[cam_idx])

            if not np.all(submat[cam_idx, 4 * idx : 4 * idx + 4, :] == -1000):
                for i in range(4):
                    plt.plot(
                        submat[cam_idx][4 * idx + i][0],
                        submat[cam_idx][4 * idx + i][1],
                        color[i],
                        10,
                    )

            if not np.all(current[cam_idx, 4 * idx : 4 * idx + 4, :] == -1000):
                for i in range(4):
                    plt.plot(
                        int(current[cam_idx][4 * idx + i][0]),
                        int(current[cam_idx][4 * idx + i][1]),
                        color_pred[i],
                        10,
                        fillstyle=None,
                    )
            plt.savefig(prefix + "camera_{}_marker_{}.png".format(cam_idx, idx))


def get_data_numpy(dirname):
    print("Getting data from {}".format(dirname))
    if isinstance(dirname, list):
        files = dirname
    elif os.path.isdir(dirname):
        onlyfiles = os.listdir(dirname)
        files = [dirname + "/" + f for f in onlyfiles]
    else:
        files = [dirname]
    poses, joint_states, observations = (None, None, None)

    print(files)

    for idx, f in enumerate(files):
        data = np.load(f)
        if idx == 0:
            observations = data["observations"]
            poses = data["poses"]
            joint_states = data["joint_states"]
        else:
            observations = np.concatenate((observations, data["observations"]))
            poses = np.concatenate((poses, data["poses"]))
            joint_states = np.concatenate((joint_states, data["joint_states"]))
    print("Loaded and accumulated data of shape: {}".format(observations.shape))
    return poses, joint_states, observations


def get_baseline(xy_0, xy_1, intrinsicsMat):
    assert xy_0.shape[1] == 2 and xy_1.shape[1] == 2
    assert xy_0.shape[0] == xy_1.shape[0]
    E, _ = cv2.findEssentialMat(xy_0, xy_1, cameraMatrix=intrinsicsMat)

    _, R_baseline, t_baseline, _, points = cv2.recoverPose(
        E, xy_0, xy_1, cameraMatrix=intrinsicsMat, distanceThresh=100000.0
    )

    assert R_baseline.shape == (3, 3)

    return R_baseline, t_baseline, points.T


def get_P(Rt, intrinsicsMat):
    R_world_in_cam, t_world_in_cam = Rt
    if R_world_in_cam.shape != (3, 3):
        R_world_in_cam = cv2.Rodrigues(R_world_in_cam)[0]
    M = np.hstack((R_world_in_cam, t_world_in_cam.reshape(-1, 1)))
    P = np.dot(intrinsicsMat, M)
    return P


def get_xyz_from_homogeneous(points):
    assert points.shape[1] == 4
    xyz_points = (points / points[:, 3].reshape(-1, 1))[:, :3]
    return xyz_points


def get_xy_from_homogeneous(points):
    assert points.shape[1] == 3
    xy_points = (points / points[:, 2].reshape(-1, 1))[:, :2]
    return xy_points


def project_xyz_to_image(P, xyz):
    xy = np.dot(P, np.hstack([xyz, np.ones((xyz.shape[0], 1))]).T).T
    xy = get_xy_from_homogeneous(xy)
    return xy


def compose(Rt1, Rt2):
    R, t = Rt1
    if R.shape != (3, 3):
        R = cv2.Rodrigues(R)[0]
    mat1 = np.eye(4)
    mat1[:3, :3] = R
    mat1[:3, 3] = t.reshape(-1)

    R, t = Rt2
    if R.shape != (3, 3):
        R = cv2.Rodrigues(R)[0]
    mat2 = np.eye(4)
    mat2[:3, :3] = R
    mat2[:3, 3] = t.reshape(-1)

    comp = np.dot(mat1, mat2)

    return cv2.Rodrigues(comp[:3, :3])[0], comp[:3, 3]


def invert(Rt):
    R_world_in_cam, t_world_in_cam = Rt
    if R_world_in_cam.shape != (3, 3):
        R_world_in_cam = cv2.Rodrigues(R_world_in_cam)[0]

    return cv2.Rodrigues(R_world_in_cam.T)[0], -R_world_in_cam.T.dot(t_world_in_cam)


def get_transform_between(A, B):
    assert len(A) == len(B)

    N = A.shape[0]
    # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - centroid_A.reshape(1, -1)
    BB = B - centroid_B.reshape(1, -1)

    H = AA.T.dot(BB)
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T.dot(U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T.dot(U.T)

    t = -R.dot(centroid_A.T) + centroid_B.T
    return R, t


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        out = n < 1e-4
        if not out:
            print(n)
        return out

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def eulerToRod(xyzrpy):
    single = len(xyzrpy.shape) == 1
    if single:
        xyzrpy = np.array([xyzrpy])
    out = np.zeros_like(xyzrpy)
    out[:, :3] = xyzrpy[:, :3]
    for i in range(len(xyzrpy)):
        out[i, 3:] = cv2.Rodrigues(eulerAnglesToRotationMatrix(xyzrpy[i, 3:]))[
            0
        ].reshape(-1)
    if single:
        return out[0]
    return out


def rodToEuler(xyzrod):
    single = len(xyzrod.shape) == 1
    if single:
        xyzrod = np.array([xyzrod])
    out = np.zeros_like(xyzrod)
    out[:, :3] = xyzrod[:, :3]
    for i in range(len(xyzrod)):
        out[i, 3:] = rotationMatrixToEulerAngles(cv2.Rodrigues(xyzrod[i, 3:])[0])
    if single:
        return out[0]
    return out


def eulerToTransformationMatrix(xyzrpy):
    single = len(xyzrpy.shape) == 1
    if single:
        xyzrpy = np.array([xyzrpy])
    out = np.zeros((xyzrpy.shape[0], 4, 4))
    out[:, :3, 3] = xyzrpy[:, :3]
    out[:, 3, 3] = 1.0
    for i in range(len(xyzrpy)):
        out[i, :3, :3] = eulerAnglesToRotationMatrix(xyzrpy[i, 3:])
    if single:
        return out[0]
    return out


def transformationMatrixtoEuler(transmat):
    single = len(transmat.shape) == 2
    if single:
        transmat = np.array([transmat])
    out = np.zeros((transmat.shape[0], 6))
    out[:, :3] = transmat[:, :3, 3]
    for i in range(len(transmat)):
        out[i, 3:] = rotationMatrixToEulerAngles(transmat[i, :3, :3])
    if single:
        return out[0]
    return out


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = (
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    )
    return rotation_matrix  # Nx4x4


def Rt_to_transform(Rt):
    R, t = Rt
    if R.shape != (3, 3):
        R = cv2.Rodrigues(R)[0]
    mat1 = np.eye(4)
    mat1[:3, :3] = R
    mat1[:3, 3] = t.reshape(-1)
    return t3d.Transform(matrix=mat1)


def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r / 2.0), np.sin(p / 2.0), np.sin(y / 2.0)
    cr, cp, cy = np.cos(r / 2.0), np.cos(p / 2.0), np.cos(y / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]  # yapf: disable


def create_solver(dh_params_torch, base_link):
    chain_obj = kdl.Chain()
    chain_obj.addSegment(
        kdl.Segment(
            "base",
            kdl.Joint("base", kdl.Joint.None_),
            kdl.Frame(
                kdl.Rotation.Quaternion(
                    *euler_to_quat(
                        *rotationMatrixToEulerAngles(
                            cv2.Rodrigues(base_link[0, 3:].detach().numpy())[0]
                        )
                    )
                ),
                kdl.Vector(*base_link[0, :3].detach().numpy()),
            ),
        )
    )

    ndof = dh_params_torch.shape[0]
    for i in range(ndof):
        chain_obj.addSegment(
            kdl.Segment(
                "{}".format(i),
                kdl.Joint(kdl.Joint.RotZ),
                kdl.Frame.DH(
                    dh_params_torch[i, 2].item(),
                    dh_params_torch[i, 3].item(),
                    dh_params_torch[i, 1].item(),
                    dh_params_torch[i, 0].item(),
                ),
            )
        )

    class URDFChainFake(object):
        def __init__(self):
            self.kdl_chain = chain_obj
            self.limits_lower_np = np.array([-3.14159265359] * ndof)
            self.limits_upper_np = np.array([3.14159265359] * ndof)
            self.limits_lower = kdl.JntArray().FromArray(self.limits_lower_np)
            self.limits_upper = kdl.JntArray().FromArray(self.limits_upper_np)
            self.base_name = "-1"
            self.tip_name = str(ndof - 1)
            self.num_dof = ndof
            self.joint_names = ["{}".format(i) for i in range(6)]

            self.joint_states = kdl.JntArray(self.num_dof)

        def set_joint_states(self, joint_states_np):
            self.joint_states.SetArray(joint_states_np)

    return LimbSolver(URDFChainFake())


def compute_initialization_from_observations(observations, fx, fy, cx, cy):
    # intrinsics
    # if "real" in env_id:
    #     fx = 904.9872633556749
    #     fy = 905.0163689345669
    #     cx = 647.189040388425
    #     cy = 351.66629230661107
    # elif "pybullet" in env_id:
    #     fx = 3280.20722537
    #     fy = 3280.49234974
    #     cx = 968.0
    #     cy = 608.0
    # else:
    #     raise ValueError("Unknown environment")

    ndof = 6

    # Camera Intrinsics Matrix
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    merge_cameras = np.concatenate(
        [observations[:, 0, :, :], observations[:, 1, :, :]], axis=-1
    )
    all_points = merge_cameras.reshape(-1, 4)
    pixel_cor = all_points[np.sum(all_points == -1000, axis=1) == 0]

    R_baseline, t_baseline, triangulated_points = get_baseline(
        pixel_cor[:, :2], pixel_cor[:, 2:], K
    )

    P_0 = get_P((np.eye(3), np.zeros((3, 1))), K)
    P_1 = get_P((R_baseline, t_baseline), K)

    reprojected_0 = get_xy_from_homogeneous(P_0.dot(triangulated_points.T).T)
    reprojected_1 = get_xy_from_homogeneous(P_1.dot(triangulated_points.T).T)
    print(
        "Cam 0 Reprojection Error: {}".format(
            ((reprojected_0 - pixel_cor[:, :2]) ** 2).mean()
        )
    )
    print(
        "Cam 1 Reprojection Error: {}".format(
            ((reprojected_1 - pixel_cor[:, 2:]) ** 2).mean()
        )
    )

    # xyz_points = get_xyz_from_homogeneous(triangulated_points)

    triangulated_points = []
    observed_in_both = []
    for i in range(observations.shape[0]):
        observed_in_cam_0 = (observations[i, 0, :, :] == -1000).sum(1) == 0
        observed_in_cam_1 = (observations[i, 1, :, :] == -1000).sum(1) == 0
        both = observed_in_cam_0 * observed_in_cam_1
        if sum(both) == 0:
            triangulated_points.append(None)
            observed_in_both.append(both)
            continue
        points_3d = cv2.triangulatePoints(
            P_0, P_1, observations[i, 0, :].T, observations[i, 1, :].T
        ).T
        points_3d = get_xyz_from_homogeneous(points_3d)
        triangulated_points.append(points_3d)
        observed_in_both.append(both)

    poses = [None] * observations.shape[0]

    for i in range(observations.shape[0]):
        if triangulated_points[i] is not None:
            poses[i] = (
                np.array([0.0, 0.0, 0.0]),
                np.mean(triangulated_points[i][observed_in_both[i]], axis=0),
            )
            break
    for i in range(0, observations.shape[0]):
        if poses[i] is None:
            for j in range(0, observations.shape[0]):
                both_pairs = observed_in_both[i] * observed_in_both[j]
                if poses[j] is not None and sum(both_pairs) >= 3:
                    (R, t) = get_transform_between(
                        triangulated_points[j][both_pairs],
                        triangulated_points[i][both_pairs],
                    )

                    new_R, new_t = compose((R, t), poses[j])
                    poses[i] = (new_R, new_t)
                    break

    for i in range(observations.shape[0] - 1, 1, -1):
        if poses[i] is None:
            for j in range(observations.shape[0] - 1, i, -1):
                both_pairs = observed_in_both[i] * observed_in_both[j]
                if poses[j] is not None and sum(both_pairs) >= 3:
                    (R, t) = get_transform_between(
                        triangulated_points[j][both_pairs],
                        triangulated_points[i][both_pairs],
                    )

                    new_R, new_t = compose((R, t), poses[j])
                    poses[i] = (new_R, new_t)
                    break

    valid_poses = np.array([idx for idx, pose in enumerate(poses) if pose is not None])

    cam_int = np.array([[fx, fy, cx, cy], [fx, fy, cx, cy]])

    cam_ext = np.zeros((2, 6))
    cam_ext[0, 3:] = 1e-5
    cam_ext[1, 3:] = cv2.Rodrigues(R_baseline)[0].reshape(-1)
    cam_ext[0, :3] = 1e-5
    cam_ext[1, :3] = t_baseline.reshape(-1)

    states = np.zeros((len(valid_poses), 6))
    for i in range(len(valid_poses)):
        states[i] = 1e-5
        if poses[valid_poses[i]] is not None:
            states[i, :3] = poses[valid_poses[i]][1].reshape(-1)
            states[i, 3:] = poses[valid_poses[i]][0].reshape(-1)
    states[0, 3:] = 1e-5

    rels = np.zeros((observations.shape[2], 3))

    dh_params = np.zeros((ndof, 4))

    base_link = np.ones((1, 6)) * 1e-3

    return ((cam_int, cam_ext, rels, dh_params, base_link), states, valid_poses)


def get_pixel_error(obs1, obs2):
    return ((obs1 - obs2) ** 2).mean()


def get_pixel_error_remove_unobserved(ground_truth, reproj):
    observed_in_cam = (ground_truth == -1000).sum(1) == 0

    sub_gt = ground_truth[observed_in_cam]
    sub_reproj = reproj[observed_in_cam]

    return get_pixel_error(sub_gt, sub_reproj)


def remove_redundant_entries(arr, tol=1e-2):
    arr = arr.copy()
    i = 0
    while i < len(arr):
        remove = False
        for j in range(i):
            if np.linalg.norm(arr[i] - arr[j]) < tol:
                remove = True
                break

        if remove:
            arr = np.delete(arr, i, axis=0)
        else:
            i += 1
    return arr


def mean_std_err(arr, axis=0):
    return arr.mean(axis=axis), arr.std(axis=axis) / np.sqrt(arr.shape[axis])
