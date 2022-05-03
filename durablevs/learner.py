from __future__ import print_function

import cv2
import numpy as np
from builtins import range

import torch
from durablevs.utils import (
    Rt_to_transform,
    angle_axis_to_rotation_matrix,
    compose,
    create_solver,
    invert,
)


class Learner(object):
    """The Learner allows for learning the camera parameters, object structure, and
    robot kinematic parameters from data.

    Parameters
    ----------
    cam_int : np.ndarray
        (num_cameras, 4) shaped array where each row contains the focal point in
        x and y and the principal point in x and y
    cam_ext : np.ndarray
        (num_cameras, 6) shaped array where each row contains the translation of the
        camera in the first three entries and the rotation of the camera in last three
        entries (in Rodrigues vector form).
    relative_poses : np.ndarray
        (num_markers, 3) shaped array where each row contains the 3D coordinates of a feature
        in the end-effector reference frame.
    dh_params : np.ndarray
        (num_markers, 4) shaped array where each row contains the 4 DH params of each of the joints
        i.e. theta_offset, d, r, alpha. See https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Denavit%E2%80%93Hartenberg_matrix.
    base_link : np.ndarray
        (1, 6) shaped array where the first 3 elements are the translation of the base frame
        and the last three elements are the rotation in Rodrigues vector form.
    """

    def __init__(self, cam_int, cam_ext, relative_poses, dh_params, base_link):
        self.cam_int = torch.tensor(cam_int, requires_grad=True)
        self.cam_ext = torch.tensor(cam_ext, requires_grad=True)
        self.relative_poses = torch.tensor(relative_poses, requires_grad=True)
        self.dh_params = torch.tensor(dh_params, requires_grad=True)
        self.base_link = torch.tensor(base_link, requires_grad=True)

        self._printing = True

    def set_printing(self, val):
        self._printing = val

    def set_relative_poses(self, relative_poses):
        self.relative_poses = torch.tensor(relative_poses, requires_grad=True)

    def save_to_file(self, filename):
        np.savez_compressed(
            filename,
            cam_int=self.cam_int.detach().numpy(),
            cam_ext=self.cam_ext.detach().numpy(),
            relative_poses=self.relative_poses.detach().numpy(),
            dh_params=self.dh_params.detach().numpy(),
            base_link=self.base_link.detach().numpy(),
        )

    @classmethod
    def make_from_file(cls, filename):
        saved_data = np.load(filename)
        return Learner(
            cam_int=saved_data["cam_int"],
            cam_ext=saved_data["cam_ext"],
            relative_poses=saved_data["relative_poses"],
            dh_params=saved_data["dh_params"],
            base_link=saved_data["base_link"],
        )

    def project(self, states):
        """Uses the structure and camera parameters to infer the pixel coordinates of the features given
        the end-effector poses.
        """
        num_cameras = self.cam_ext.shape[0]
        projections = []
        for c in range(num_cameras):
            rels_ = torch.cat(
                (
                    self.relative_poses,
                    torch.ones((self.relative_poses.shape[0], 1), dtype=torch.float64),
                ),
                1,
            )
            points = torch.einsum("noi, mi -> nmo", states, rels_).reshape(-1, 4)

            cam_matrix = torch.zeros(3, 4, dtype=torch.float64)
            cam_matrix[0, 0] = self.cam_int[c, 0]
            cam_matrix[1, 1] = self.cam_int[c, 1]
            cam_matrix[0, 2] = self.cam_int[c, 2]
            cam_matrix[1, 2] = self.cam_int[c, 3]
            cam_matrix[2, 2] = 1.0

            cam_rot_mat = angle_axis_to_rotation_matrix(
                torch.stack([self.cam_ext[c, 3:]])
            )[0][:3, :3]
            cam_ext_h = torch.cat(
                (
                    torch.cat((cam_rot_mat, self.cam_ext[c, :3].unsqueeze(1)), 1),
                    torch.DoubleTensor([[0.0, 0.0, 0.0, 1.0]]),
                ),
                0,
            )
            uv = cam_matrix.mm(cam_ext_h.mm(points.t()))
            projection = ((uv[:2] / uv[2]).t()).reshape(
                states.shape[0], self.relative_poses.shape[0], 2
            )
            projections.append(projection)

        return torch.stack(projections, 1)

    def fk(self, jps):
        """Uses kinematic structure to infer cartesian pose of the end-effector given the joint positions.
        """
        n = jps.shape[0]
        theta = jps
        theta_offset, d, r, alpha = self.dh_params.repeat(n, 1, 1).permute(2, 0, 1)

        rots = self.base_link[0, 3:].reshape(-1, 3)
        out = angle_axis_to_rotation_matrix(rots)
        out[..., :3, 3] = self.base_link[0, :3]
        base_frame = out.repeat(n, 1, 1).permute(1, 2, 0)

        ndof = self.dh_params.shape[0]
        dh_mat = torch.stack(
            [
                torch.cos(theta + theta_offset),
                -torch.sin(theta + theta_offset) * torch.cos(alpha),
                torch.sin(theta + theta_offset) * torch.sin(alpha),
                r * torch.cos(theta + theta_offset),
                torch.sin(theta + theta_offset),
                torch.cos(theta + theta_offset) * torch.cos(alpha),
                -torch.cos(theta + theta_offset) * torch.sin(alpha),
                r * torch.sin(theta + theta_offset),
                torch.zeros((n, ndof), dtype=torch.float64),
                torch.sin(alpha),
                torch.cos(alpha),
                d,
                torch.zeros((n, ndof), dtype=torch.float64),
                torch.zeros((n, ndof), dtype=torch.float64),
                torch.zeros((n, ndof), dtype=torch.float64),
                torch.ones((n, ndof), dtype=torch.float64),
            ]
        ).reshape(4, 4, n, ndof)

        T = base_frame
        for i in range(ndof):
            T = torch.einsum("abc, bdc -> adc", T, dh_mat[:, :, :, i])
        return T.permute(2, 0, 1)

    def loss_image(self, predicted, target, mask=None):
        """Masked image space pixel norm."""
        mask_non_visible = (target != -1000).type_as(target)
        res = (target - predicted) * mask_non_visible
        if mask is not None:
            res *= mask
        loss = (res ** 2).mean()
        return loss

    def loss_cart(self, predicted, target):
        """Cartesian pose norm."""
        num_samples = predicted.shape[0]
        res = (predicted[:, :3, 3] - target[:, :3, 3]) ** 2
        res_rot = (predicted[:, :3, :3] - target[:, :3, :3]) ** 2
        loss = res.mean() + res_rot.mean()
        return loss / num_samples

    def learn(
        self,
        inp,
        target,
        func,
        params_to_learn,
        learn_input=False,
        niter=100,
        learning_rate=1.0,
        description="",
        batch=False,
        num_batches=5,
        mask=None,
    ):
        if batch:
            split_inp = np.array_split(inp, num_batches)
            split_target = np.array_split(target, num_batches)
            split_iterations = int(niter / num_batches)
            for i in range(num_batches):
                self.learn(
                    split_inp[i],
                    split_target[i],
                    func,
                    params_to_learn,
                    learn_input=learn_input,
                    niter=split_iterations,
                    learning_rate=learning_rate,
                    description="Batch {} - ".format(i) + description,
                    batch=False,
                )
            return self.learn(
                inp,
                target,
                func,
                params_to_learn,
                learn_input=learn_input,
                niter=niter,
                learning_rate=learning_rate,
                description="All Data Batch - " + description,
                batch=False,
            )

        target_t = torch.tensor(target)
        input_t = torch.tensor(inp, requires_grad=learn_input)
        if learn_input:
            params_to_learn += [input_t]
        optimizer = torch.optim.LBFGS(params_to_learn, lr=learning_rate)

        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                if func == "fk":
                    predicted = self.fk(input_t)
                elif func == "cam":
                    predicted = self.project(self.pose2rot(input_t))
                elif func == "all":
                    predicted = self.project(self.fk(input_t))

                if func == "fk":
                    loss = self.loss_cart(predicted, target_t)
                else:
                    loss = self.loss_image(predicted, target_t, mask=mask)
                loss.backward()
                if self._printing:
                    print(description + " It. {}/{}, Loss: {}".format(itr, niter, loss))
                return loss

            optimizer.step(closure)

        return input_t.detach().numpy()

    # Learners
    def learn_cameras(
        self,
        cartesian_states,
        image,
        niter=100,
        learning_rate=1.0,
        batch=False,
        num_batches=5,
    ):
        return self.learn(
            cartesian_states,
            image,
            "cam",
            [self.cam_ext, self.relative_poses, self.cam_int],
            learn_input=True,
            niter=niter,
            learning_rate=learning_rate,
            description="Camera",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_cameras_from_joint_and_image(
        self,
        joint_states,
        image,
        niter=100,
        learning_rate=1.0,
        batch=False,
        num_batches=5,
    ):
        return self.learn(
            joint_states,
            image,
            "all",
            [self.cam_ext, self.cam_int],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="Camera from Joints and Image",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_state_from_image(
        self,
        cartesian_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):

        return self.learn(
            cartesian_states,
            image,
            "cam",
            [],
            learn_input=True,
            niter=niter,
            learning_rate=learning_rate,
            description="State from Image",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_fk(
        self,
        joint_states,
        cartesian_states,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):
        cartesian_states_mat = np.zeros((cartesian_states.shape[0], 3, 4))
        for i in range(cartesian_states.shape[0]):
            cartesian_states_mat[i, :3, :3] = cv2.Rodrigues(cartesian_states[i, 3:])[0]
            cartesian_states_mat[i, :3, 3] = cartesian_states[i, :3]
        return self.learn(
            joint_states,
            cartesian_states_mat,
            "fk",
            [self.dh_params, self.base_link],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="FK",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_baselink(
        self,
        joint_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):
        return self.learn(
            joint_states,
            image,
            "all",
            [self.base_link],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="Base Link",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_fk_e2e(
        self,
        joint_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):
        return self.learn(
            joint_states,
            image,
            "all",
            [self.dh_params, self.base_link],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="FK E2E",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_joints_from_state(
        self,
        joint_states,
        cartesian_states,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):
        cartesian_states_mat = np.zeros((cartesian_states.shape[0], 3, 4))
        for i in range(cartesian_states.shape[0]):
            cartesian_states_mat[i, :3, :3] = cv2.Rodrigues(cartesian_states[i, 3:])[0]
            cartesian_states_mat[i, :3, 3] = cartesian_states[i, :3]
        return self.learn(
            joint_states,
            cartesian_states_mat,
            "fk",
            [],
            learn_input=True,
            niter=niter,
            learning_rate=learning_rate,
            description="Joints from State",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_all(
        self,
        joint_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):
        return self.learn(
            joint_states,
            image,
            "all",
            [
                self.cam_ext,
                self.relative_poses,
                self.dh_params,
                self.cam_int,
                self.base_link,
            ],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="All",
            batch=batch,
            num_batches=num_batches,
        )

    def learn_relative_poses_cart(
        self,
        cartesian_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
        mask=None,
    ):
        return self.learn(
            cartesian_states,
            image,
            "cam",
            [self.relative_poses],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="Relative Poses Cartesian",
            batch=batch,
            num_batches=num_batches,
            mask=mask,
        )

    def learn_relative_poses_joint(
        self,
        joint_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
        mask=None,
    ):
        return self.learn(
            joint_states,
            image,
            "all",
            [self.relative_poses],
            learn_input=False,
            niter=niter,
            learning_rate=learning_rate,
            description="Relative Poses Joints",
            batch=batch,
            num_batches=num_batches,
            mask=mask,
        )

    def learn_joints_from_image(
        self,
        joint_states,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
    ):
        return self.learn(
            joint_states,
            image,
            "all",
            [],
            learn_input=True,
            niter=niter,
            learning_rate=learning_rate,
            description="Joints from Image",
            batch=batch,
            num_batches=num_batches,
        )

    def loss_joint_actions_image(
        self, predicted, jps, target, actions, weight=1000.0, which="12"
    ):
        mask = (target != -1000).type_as(target)
        res = (target - predicted) * mask
        part1 = (res ** 2).mean()

        jps_diff = jps[1:] - jps[:-1]
        part2 = ((jps_diff - actions) ** 2).mean()

        if "12" == which:
            return part1 + part2 * weight
        elif "1" == which:
            return part1
        else:
            return part2

    def loss_joint_actions_cart(
        self, predicted, jps, target, actions, weight=1000.0, which="12"
    ):
        num_samples = predicted.shape[0]
        res = (predicted[:, :3, 3] - target[:, :3, 3]) ** 2
        res_rot = (predicted[:, :3, :3] - target[:, :3, :3]) ** 2
        part1 = res.mean() + res_rot.mean()
        part1 /= num_samples

        jps_diff = jps[1:] - jps[:-1]
        part2 = ((jps_diff - actions) ** 2).mean()

        if "12" == which:
            return part1 + part2 * weight
        elif "1" == which:
            return part1
        else:
            return part2

    def learn_fk_joint_actions(
        self,
        joint_states,
        actions,
        cartesian_states,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
        weight=1000.0,
        learn_joints=True,
        which="12",
    ):

        if batch:
            split_inp = np.array_split(joint_states, num_batches)
            split_actions = np.array_split(actions, num_batches)
            split_target = np.array_split(cartesian_states, num_batches)
            split_iterations = niter / num_batches
            for i in range(len(split_inp)):
                self.learn_fk_joint_actions(
                    split_inp[i],
                    split_actions[i],
                    split_target[i],
                    niter=split_iterations,
                    learning_rate=learning_rate,
                    batch=False,
                    weight=weight,
                    learn_joints=learn_joints,
                    which=which,
                )
            return self.learn_fk_joint_actions(
                joint_states,
                actions,
                cartesian_states,
                niter=niter,
                learning_rate=learning_rate,
                batch=False,
                weight=weight,
                learn_joints=learn_joints,
                which=which,
            )

        cartesian_states_mat = np.zeros((cartesian_states.shape[0], 3, 4))
        for i in range(cartesian_states.shape[0]):
            cartesian_states_mat[i, :3, :3] = cv2.Rodrigues(cartesian_states[i, 3:])[0]
            cartesian_states_mat[i, :3, 3] = cartesian_states[i, :3]

        cartesian_states_mat = torch.tensor(cartesian_states_mat)
        joint_states = torch.tensor(joint_states, requires_grad=True)
        actions = torch.tensor(actions[:-1])
        params_to_learn = [self.dh_params, self.base_link]
        if learn_joints:
            params_to_learn += [joint_states]
        optimizer = torch.optim.LBFGS(params_to_learn, lr=learning_rate)

        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                predicted = self.fk(joint_states)
                loss = self.loss_joint_actions_cart(
                    predicted,
                    joint_states,
                    cartesian_states_mat,
                    actions,
                    weight=weight,
                    which=which,
                )
                loss.backward()
                if self._printing:
                    print("FK Actions It. {}/{}, Loss: {}".format(itr, niter, loss))
                return loss

            optimizer.step(closure)

        return joint_states.detach().numpy()

    def learn_all_joint_actions(
        self,
        joint_states,
        actions,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
        weight=1000.0,
        learn_joints=True,
        which="12",
    ):

        if batch:
            split_inp = np.array_split(joint_states, num_batches)
            split_actions = np.array_split(actions, num_batches)
            split_target = np.array_split(image, num_batches)
            split_iterations = niter / num_batches
            for i in range(len(split_inp)):
                self.learn_all_joint_actions(
                    split_inp[i],
                    split_actions[i],
                    split_target[i],
                    niter=split_iterations,
                    learning_rate=learning_rate,
                    batch=False,
                    weight=weight,
                    learn_joints=learn_joints,
                    which=which,
                )
            return self.learn_all_joint_actions(
                joint_states,
                actions,
                image,
                niter=niter,
                learning_rate=learning_rate,
                batch=False,
                weight=weight,
                learn_joints=learn_joints,
                which=which,
            )

        observations = torch.tensor(image)
        joint_states = torch.tensor(joint_states, requires_grad=True)
        actions = torch.tensor(actions[:-1])
        params_to_learn = [
            self.cam_ext,
            self.relative_poses,
            self.dh_params,
            self.base_link,
        ]
        if learn_joints:
            params_to_learn += [joint_states]
        optimizer = torch.optim.LBFGS(params_to_learn, lr=learning_rate)

        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                predicted = self.project(self.fk(joint_states))
                loss = self.loss_joint_actions_image(
                    predicted,
                    joint_states,
                    observations,
                    actions,
                    weight=weight,
                    which=which,
                )
                loss.backward()
                if self._printing:
                    print("All Actions It. {}/{}, Loss: {}".format(itr, niter, loss))
                return loss

            optimizer.step(closure)

        return joint_states.detach().numpy()

    def loss_cart_actions_image(
        self, predicted, carts, target, actions, weight=1000.0, which="12"
    ):
        mask = (target != -1000).type_as(target)
        res = (target - predicted) * mask
        part1 = (res ** 2).mean()

        res_tot = 0.0
        res_rot_tot = 0.0
        for i in range(carts.shape[0] - 1):
            next_pose = torch.mm(carts[i], actions[i])
            res = (next_pose[:3, 3] - carts[i + 1, :3, 3]) ** 2
            res_rot = (next_pose[:3, :3] - carts[i + 1, :3, :3]) ** 2
            res_tot += res.mean()
            res_rot_tot += res_rot.mean()

        part2 = res_tot + res_rot_tot
        if "12" == which:
            return part1 + part2 * weight
        elif "1" == which:
            return part1
        else:
            return part2

    def pose_loss(self, states1, states2):
        res = (states1[:, :3, 3] - states2[:, :3, 3]) ** 2
        res_rot = (states1[:, :3, :3] - states2[:, :3, :3]) ** 2
        return res.mean() + res_rot.mean()

    def scale_and_transform(self, states, s, t1, t2):
        rots1 = t1[:, 3:].reshape(-1, 3)
        out1 = angle_axis_to_rotation_matrix(rots1)
        out1[..., :3, 3] = t1[..., :3]
        out1 = out1[0]
        out1[3, 3] = s

        rots2 = t2[:, 3:].reshape(-1, 3)
        out2 = angle_axis_to_rotation_matrix(rots2)
        out2[..., :3, 3] = t2[..., :3]
        out2 = out2[0]
        out2[3, 3] = s

        temp = torch.einsum("ij, ajk -> aik", out1, states)
        result = torch.einsum("aij, jk -> aik", temp, torch.inverse(out2))
        return result

    def pose2rot(self, states):
        """ Convert pose to associated rotation matrix
            s.shape = (n_poses, 6), where the 6 positions are x, y, z, phi, theta, psi for each pose
            Output shape is (n_poses, 4, 4)"""
        rots = states[:, 3:].reshape(-1, 3)
        out = angle_axis_to_rotation_matrix(rots)
        out[..., :3, 3] = states[..., :3]
        return out

    def end_to_end(self, joint_states, image):
        observations = torch.tensor(image)
        jps = torch.tensor(joint_states)
        predicted = self.project(self.fk(jps))
        loss = self.loss_image(predicted, observations)
        loss.backward()
        return loss.item(), predicted.detach().numpy()

    def cart_to_end(self, carts, image):
        observations = torch.tensor(image)
        carts = torch.tensor(carts)
        predicted = self.project(self.pose2rot(carts))
        loss = self.loss_image(predicted, observations)
        loss.backward()
        return loss.item(), predicted.detach().numpy()

    def eval_all(self, joint_states):
        jps = torch.tensor(joint_states)
        predicted = self.project(self.fk(jps))
        return predicted.detach().numpy()

    def get_gradients(self, joint_states, image):
        observations = torch.tensor(image)
        jps = torch.tensor(joint_states)
        predicted = self.project(self.fk(jps))
        loss = self.loss_image(predicted, observations)
        loss.backward()
        return (
            self.cam_int.grad,
            self.cam_ext.grad,
            self.base_link.grad,
            self.dh_params.grad,
            self.relative_poses.grad,
        )

    def zero_gradients(self):
        self.cam_int.data.zero_()
        self.cam_ext.data.zero_()
        self.relative_poses.data.zero_()
        self.dh_params.data.zero_()
        self.base_link.data.zero_()

    def pnp(self, observation_matrix):
        Rs, ts = [], []
        found = False
        num_cameras = observation_matrix.shape[0]

        relative_poses = self.relative_poses.detach().numpy()

        def get_observed_points(l):
            return (l == -1000).sum(1) == 0

        for idx in range(num_cameras):
            try:
                cam_pose = (
                    self.cam_ext[idx, 3:].detach().numpy(),
                    self.cam_ext[idx, :3].detach().numpy(),
                )

                fx = self.cam_int[idx, 0].item()
                fy = self.cam_int[idx, 1].item()
                cx = self.cam_int[idx, 2].item()
                cy = self.cam_int[idx, 3].item()
                K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

                seen_in_cam = get_observed_points(observation_matrix[idx])
                _, Ro, to = cv2.solvePnP(
                    relative_poses[seen_in_cam],
                    observation_matrix[idx][seen_in_cam].copy(),
                    K,
                    np.array([0.0, 0.0, 0.0, 0.0]),
                )
                R, t = compose(invert(cam_pose), (Ro.reshape(-1), to.reshape(-1)))
                Rs.append(R)
                ts.append(t)
                found = True
            except Exception:
                pass

        if not found:
            raise ValueError("not visible")

        Ro = np.array(Rs).mean(0)
        to = np.array(ts).mean(0)
        curr_pose = np.zeros((1, 6))
        curr_pose[0, :3] = to.reshape(-1)
        curr_pose[0, 3:] = Ro.reshape(-1)
        return curr_pose

    def ik(self, curr_pose, seed_joint_positions, solver=None):
        assert False
        # if solver is None:
        #     solver = create_solver(self.dh_params, self.base_link)
        # kdl_target = transform_to_kdl_frame(
        #     Rt_to_transform((curr_pose[0, 3:], curr_pose[0, :3]))
        # )
        # joints = solver.calc_ik(kdl_target, seed_joint_positions)[1].Array().T
        # return joints

    def ik_get_many(self, observations, range_low, range_high, num_sols=100):
        target_pose = self.pnp(observations)
        ik_sols_target = np.zeros((num_sols, 6))
        for i in range(num_sols):
            seed = np.random.uniform(range_low, range_high)
            ik_sols_target[i] = self.ik(target_pose, seed)[0]
        return ik_sols_target

    def learn_transformation(
        self, states1, states2, niter=100, learning_rate=0.2, s=None, T1=None, T2=None
    ):
        if s is None:
            s = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
        else:
            s = torch.tensor(s, requires_grad=True, dtype=torch.float64)

        if T1 is None:
            T1 = torch.tensor(
                [[1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]],
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            T1 = torch.tensor(T1, requires_grad=True, dtype=torch.float64)

        if T2 is None:
            T2 = torch.tensor(
                [[1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]],
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            T2 = torch.tensor(T2, requires_grad=True, dtype=torch.float64)

        optimizer = torch.optim.LBFGS([s, T1, T2], lr=learning_rate)
        states1 = torch.tensor(states1, requires_grad=True)
        states2 = torch.tensor(states2, requires_grad=True)

        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                loss = self.pose_loss(
                    self.scale_and_transform(self.pose2rot(states1), s, T1, T2),
                    self.pose2rot(states2),
                )
                loss.backward()
                if self._printing:
                    print("Transformation It. {}/{}, Loss: {}".format(itr, niter, loss))
                return loss

            optimizer.step(closure)

        return (
            s.item(),
            T1.detach().numpy(),
            T2.detach().numpy(),
            self.scale_and_transform(self.pose2rot(states1), s, T1, T2)
            .detach()
            .numpy(),
            self.pose2rot(states2).detach().numpy(),
        )

    def learn_cameras_cart_actions(
        self,
        cartesian_states,
        actions,
        image,
        niter=100,
        learning_rate=1.0,
        batch=False,
        num_batches=5,
        weight=1000.0,
        which="12",
    ):

        if batch:
            split_inp = np.array_split(cartesian_states, num_batches)
            split_actions = np.array_split(actions, num_batches)
            split_target = np.array_split(image, num_batches)
            split_iterations = niter / num_batches
            for i in range(len(split_inp)):
                self.learn_cameras_cart_actions(
                    split_inp[i],
                    split_actions[i],
                    split_target[i],
                    niter=split_iterations,
                    learning_rate=learning_rate,
                    batch=False,
                    weight=weight,
                    which=which,
                )
            return self.learn_cameras_cart_actions(
                cartesian_states,
                actions,
                image,
                niter=niter,
                learning_rate=learning_rate,
                batch=False,
                weight=weight,
                which=which,
            )

        observations = torch.tensor(image)
        cartesian_states = torch.tensor(cartesian_states, requires_grad=True)
        actions = torch.tensor(actions[:-1])

        params_to_learn = [self.cam_ext, self.relative_poses, cartesian_states]
        optimizer = torch.optim.LBFGS(params_to_learn, lr=learning_rate)
        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                predicted = self.project(self.pose2rot(cartesian_states))
                loss = self.loss_cart_actions_image(
                    predicted,
                    self.pose2rot(cartesian_states),
                    observations,
                    actions,
                    weight=weight,
                    which=which,
                )
                loss.backward()
                if self._printing:
                    print(
                        "Cameras Actions It. {}/{}, Loss: {}".format(itr, niter, loss)
                    )
                return loss

            optimizer.step(closure)
        return cartesian_states.detach().numpy()

    def loss_noisy_joints_image(self, predicted, jps, target, jps_gt, weight=1000.0):
        mask = (target != -1000).type_as(target)
        res = (target - predicted) * mask
        part1 = (res ** 2).mean()

        part2 = ((jps - jps_gt) ** 2).mean()

        return part1 + part2 * weight

    def learn_all_noisy_joints(
        self,
        joint_states,
        joints_gt,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
        weight=1000.0,
        learn_joints=True,
    ):

        observations = torch.tensor(image)
        joints_gt = torch.tensor(joints_gt)
        joint_states = torch.tensor(joint_states, requires_grad=True)

        params_to_learn = [
            self.cam_ext,
            self.relative_poses,
            self.dh_params,
            self.base_link,
        ]
        if learn_joints:
            params_to_learn += [joint_states]
        optimizer = torch.optim.LBFGS(params_to_learn, lr=learning_rate)

        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                predicted = self.project(self.fk(joint_states))
                loss = self.loss_noisy_joints_image(
                    predicted, joint_states, observations, joints_gt, weight=weight
                )
                loss.backward()
                if self._printing:
                    print(
                        "Noisy Joints All It. {}/{}, Loss: {}".format(itr, niter, loss)
                    )
                return loss

            optimizer.step(closure)

        return joint_states.detach().numpy()

    def loss_noisy_cart_image(self, predicted, carts, target, carts_gt, weight=1000.0):
        mask = (target != -1000).type_as(target)
        res = (target - predicted) * mask
        part1 = (res ** 2).mean()

        res = (carts[:, :3, 3] - carts_gt[:, :3, 3]) ** 2
        res_rot = (carts[:, :3, :3] - carts_gt[:, :3, :3]) ** 2
        part2 = res.mean() + res_rot.mean()

        return part1 + part2 * weight

    def learn_cameras_noisy_cart(
        self,
        cartesian_states,
        cartesian_states_gt,
        image,
        niter=100,
        learning_rate=0.2,
        batch=False,
        num_batches=5,
        weight=1000.0,
        learn_carts=True,
    ):

        observations = torch.tensor(image)
        cartesian_states = torch.tensor(cartesian_states, requires_grad=True)
        cartesian_states_gt = torch.tensor(cartesian_states_gt)

        params_to_learn = [self.cam_ext, self.relative_poses]
        if learn_carts:
            params_to_learn += [cartesian_states]
        optimizer = torch.optim.LBFGS(params_to_learn, lr=learning_rate)

        for itr in range(niter):

            def closure():
                optimizer.zero_grad()
                predicted = self.project(self.pose2rot(cartesian_states))
                loss = self.loss_noisy_cart_image(
                    predicted,
                    self.pose2rot(cartesian_states),
                    observations,
                    self.pose2rot(cartesian_states_gt),
                    weight=weight,
                )
                loss.backward()
                if self._printing:
                    print(
                        "Noisy Cart Cameras It. {}/{}, Loss: {}".format(
                            itr, niter, loss
                        )
                    )
                return loss

            optimizer.step(closure)
        return cartesian_states.detach().numpy()
