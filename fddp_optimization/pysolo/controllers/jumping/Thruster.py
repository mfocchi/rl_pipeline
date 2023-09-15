import numpy as np
import numpy as np
# Pinocchio modules
import pinocchio as pin  # Pinocchio library
import scipy.linalg as linalg
from pysolo.controllers.utils.base_controller import BaseCtrl
from pysolo.controllers.utils.print_style import PrintStyle


#np.set_printoptions(suppress=True, precision=3, linewidth=os.get_terminal_size(0)[0])

class Thruster(BaseCtrl):
    def __init__(self, robot, x_max=0., y_max=0., z_max=0., alpha=0.5, T_landing = .5,
                 only_thruster = False, vectors_for_plots = True):
        '''
        Args:
            alpha: force scaling factor [max force actuated = alpha * max force achievable]
            dt: sampling time
            only_thruster: True if the user does not want to use the lander
            vectors_for_plot: True if the user wants to save all the variables for making plots
        '''
        super().__init__(robot)
        # Arguments
        self._x_max = x_max
        self._y_max = y_max
        self._z_max = z_max
        self._alpha= alpha
        self._Bezier_coeffs_normalized = np.array([[0.0, 0.8, 1.0, 1.0], [1.0, 1.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self._Bezier_average = 0.5*(self._Bezier_coeffs_normalized[0] + self._Bezier_coeffs_normalized[1]).mean()

        self._vectors_for_plots = vectors_for_plots

        self._jump_height = self._z_max - self._com_home[2]

        # Jacobian (ee: stack of feet)
        self.J_xyz_stack = np.empty([3 * self._nfeet, self._nqa])

        # These fields are filled by calling force_shaping()
        # T_stance: time horizon for the thrusting phase
        # T_air: time of flight
        # T_total = T_stance + T_air + 0.5 (0.5 is added for see what happen after the touch-down, can be changed)
        # samples: number of time samples between 0.0 and T_total of lenght dt
        # time_span: vector containing all the sample instants
        self.T_stance = 0.
        self.T_air = 0.
        self.T_total = T_landing # = T_stance + T_air + T_landing
        self.samples = 0
        self.sample_at_lift_off = 0
        self.sample_at_touch_down = 0
        self.time_span = np.empty(0)

        self._compute_times()

        # order: sample, foot_xyz
        self._f_ff = np.empty([self.samples,12])
        self._force_shaping()

        # These fields are filled by calling reference()
        # order: sample, xyz
        self.com_ref = np.empty([self.samples,3])
        self.comdot_ref = np.empty([self.samples,3])
        self.comddot_ref = np.empty([self.samples,3])

        # order: sample, feet, xyz
        self.feet_pos_ref = np.empty([self.samples,4,3])
        self.feet_lin_vel_ref = np.empty([self.samples,4,3])

        self._reference()

        # fb gains
        kp = 100.0
        kd = 10.0
        self._Kp = kp * np.eye(3 * self._nfeet)
        self._Kd = kd * np.eye(3 * self._nfeet)

    def _compute_times(self):
        f_z_max = self._max_force_along_z()

        self.T_air = np.sqrt(8 * self._jump_height / self._g)
        self.T_air = int(self.T_air/self._dt) * self._dt
        self.T_stance = (self._weight * self.T_air) / (4 * f_z_max * self._Bezier_average - self._weight)
        self.T_stance = int(self.T_stance / self._dt) * self._dt


        self.sample_at_lift_off = int(np.ceil(self.T_stance / self._dt) + 1)
        self.sample_at_touch_down = self.sample_at_lift_off + int(np.ceil(self.T_air / self._dt) + 1)
        self.T_total += self.T_stance + self.T_air
        self.samples = int(np.ceil(self.T_total / self._dt) + 1)
        self.time_span = np.linspace(0.0, self.T_total, self.samples)


    def _force_shaping(self):
        # Feedforward force is a Bèzier 3rd order polynomial
        # f_ff(t) = b_0*(1-t)^3 + b_1*t*(1-t)^2 + b_2*t^2*(1-t) + b_3*t^3
        #         = Bezier_coeffs' * time_vec3(t)
        time_vec3 = lambda t: np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])

        f_z_max = self._max_force_along_z()
        f_x_max = self._max_force_along_xy(self._x_max)
        f_y_max = self._max_force_along_xy(self._y_max)

        Bezier_coeffs_x = f_x_max * self._Bezier_coeffs_normalized
        Bezier_coeffs_y = f_y_max * self._Bezier_coeffs_normalized
        Bezier_coeffs_z = f_z_max * self._Bezier_coeffs_normalized

        self._feedforward_force(Bezier_coeffs_x, Bezier_coeffs_y, Bezier_coeffs_z, time_vec3)


    def _reference(self):
        self.com_ref[0, 0:3] = self._com_home[0:3]
        self._com_ref_in_universe_frame()
        self._feet_ref_in_base_frame()


    # The output of the controller is the torque as a njoints-dimensional vector and the contact forces
    def feet_PD_controller(self, J_xyz_stack, base_orientation, feet_pos, feet_pos_ref,
                           feet_lin_vel, feet_lin_vel_ref = None):
        # stabilizes the feet position about the initial position moving with a frame centered on the CoM with axes parallel to the world frame
        Kp = self._Kp
        Kd = self._Kd
        if feet_lin_vel_ref is None:
            feet_lin_vel_ref = np.zeros_like(feet_lin_vel)
        err = feet_pos - feet_pos_ref
        errdot = feet_lin_vel - feet_lin_vel_ref

        # Compute PD feedback
        w = float(base_orientation[3])
        x = float(base_orientation[0])
        y = float(base_orientation[1])
        z = float(base_orientation[2])
        R_BU = pin.Quaternion(w, x, y, z).toRotationMatrix()
        R_BU_stack = linalg.block_diag(R_BU, R_BU, R_BU, R_BU)

        f_fb_B = np.matmul(Kp, err) + np.matmul(Kd, errdot)
        f_fb = np.matmul(R_BU_stack, f_fb_B).T
        tau_fb = -np.matmul(J_xyz_stack.T, f_fb).flatten()

        return tau_fb, f_fb

    def joint_PID_controller(self, q, qdes, qdot):
        kp = 1000
        kd = 50
        ki = 20

        tau_fb = kp*(qdes-q)-kd*qdot
        f_fb = np.zeros_like(tau_fb)

        return tau_fb, f_fb


    def compute(self, q, qdot, sample, current_time, in_air = False):
        # Execute the thrusting phase

        # Compute/update all the joints and frames
        self.forward_kinematics(q.copy(), qdot.copy(), 'base_link')

        # Get force
        f_ff = self._f_ff[sample]

        # Get Jacobians
        for idx, foot_frame in enumerate(self._feet_frames):
            J_foot = pin.getFrameJacobian(self._robot.model, self._robot.data, foot_frame,
                                          self._local_world_aligned_frame)
            self.J_xyz_stack[idx * 3:(idx + 1) * 3] = J_foot[0:3, 6:].copy()

        # Compute joint torque
        tau_ff = -np.matmul(self.J_xyz_stack.T, f_ff).flatten()
        tau_fb = np.zeros_like(tau_ff)

        # Compute feet position and velocity
        # here the vectors are stacked per columns
        # TODO: are the feet position/velocity correctly read?
        feet_pos_stack = np.zeros(12)
        feet_lin_vel_stack = np.zeros(12)

        for idx, frame_id in enumerate(self._feet_frames):
            feet_pos_stack[idx * 3:(idx + 1) * 3] = self._robot.data.oMf[frame_id].translation.transpose()  # wrt world frame
            frame = self._robot.model.frames[frame_id]
            foot_rot = self._robot.data.oMf[frame_id].rotation
            oMff = pin.SE3(foot_rot, np.zeros(3))
            iMf = frame.placement
            f_v = iMf.actInv(self._robot.data.v[frame.parent])
            feet_lin_vel_stack[idx * 3:(idx + 1) * 3] = oMff.act(f_v).linear.transpose()

        # CONTROL LAW
        # Track feet reference expressed in base frame (retrieved from com trajectory expressed in universal frame)
        # Get the reference position and linear velocity of the feet in the base frame
        p_ref = self.feet_pos_ref[sample]
        lv_ref = self.feet_lin_vel_ref[sample]

        feet_pos_ref_stack = np.hstack([p_ref[0], p_ref[1], p_ref[2], p_ref[3]])
        feet_lin_vel_ref_stack = np.hstack([lv_ref[0], lv_ref[1], lv_ref[2], lv_ref[3]])
        if not in_air:
            tau_fb, f_fb = self.feet_PD_controller(self.J_xyz_stack, q[3:7], feet_pos_stack, feet_pos_ref_stack, feet_lin_vel_stack, feet_lin_vel_ref_stack)
        else:
            tau_fb, f_fb = self.joint_PID_controller(q[7:], self._q_home[7:], qdot[6:])


        tau = tau_ff + tau_fb
        # check saturations
        tau_sat = np.clip(tau, self._tau_min, self._tau_max)
        if False in (tau == tau_sat):
            print('Thruster:')
            print(PrintStyle.WARNING + "torque saturation" + PrintStyle.ENDC)
            print('tau', tau, '\nerr', tau-tau_sat)

        if self._vectors_for_plots:# and sample != 0:
            data = self._store_all_data(q=q, qdot=qdot, com_ref=self.com_ref[sample], comdot_ref=self.comdot_ref[sample], f=f_ff+f_fb,
                                        f_ff=f_ff, f_fb=f_fb, tau=tau_sat, tau_fb=tau_fb, tau_ff=tau_ff)
        else:
            data = {'torque': tau_sat}

        #qddot = self.forward_dynamics(self._robot.model, self._robot.data, q, qdot, tau, fc = f_ff+f_fb)
        #q, qdot = self.integrate(q, qdot, qddot)

        return data#, q, qdot, qddot


    #############################
    # Methods for force shaping #
    #############################

    def _max_force_along_z(self):
        # f_max is the force exerted when the robot is in the init config
        lower_leg_disp = self._robot.model.frames[self._feet_frames[0]].placement.translation
        lower_leg_length = np.sqrt(np.matmul(lower_leg_disp.T, lower_leg_disp))

        q_FL_hip = self._q_home[8]
        q_FL_knee = self._q_home[9]
        f_z_max = self._alpha * self._tau_max[0] / lower_leg_length * abs(np.sin(q_FL_hip - q_FL_knee))

        return f_z_max


    def _max_force_along_xy(self, displacement):
        # f_x_max is computed approximating the the lift off x-position of the CoM with the initial position of the CoM
        f_max = self._alpha * displacement * self._mass / (4* self._Bezier_average * self.T_stance ** 2)
        return f_max

    def _feedforward_force(self, Bezier_coeffs_x, Bezier_coeffs_y, Bezier_coeffs_z, time_vec3):
        # assign to all the feet the same force (f_x, 0, f_z)
        # return a matrix in which the i-th columns contain the force applied by the contact points of the robot on the
        # environment at sampling time i
        f_x = 0.0
        f_y = 0.0
        f_z = 0.0

        for idx, current_time in enumerate(self.time_span):
            if current_time < self.T_stance / 2:
                # f_z continuously increases
                t = 2*current_time / self.T_stance
                f_x = np.dot(Bezier_coeffs_x[0], time_vec3(t))
                f_y = np.dot(Bezier_coeffs_y[0], time_vec3(t))
                f_z = np.dot(Bezier_coeffs_z[0], time_vec3(t))


            elif self.T_stance / 2 <= current_time < self.T_stance:
                # f_z continuously decreases
                t = 2 * current_time / self.T_stance -1
                f_x = np.dot(Bezier_coeffs_x[1], time_vec3(t))
                f_y = np.dot(Bezier_coeffs_y[1], time_vec3(t))
                f_z = np.dot(Bezier_coeffs_z[1], time_vec3(t))

            else:
                f_x = 0.0
                f_y = 0.0
                f_z = 0.0


            self._f_ff[idx, 0]  = f_x
            self._f_ff[idx, 3]  = f_x
            self._f_ff[idx, 6]  = f_x
            self._f_ff[idx, 9]  = f_x

            self._f_ff[idx, 1]  = f_y
            self._f_ff[idx, 4]  = f_y
            self._f_ff[idx, 7]  = f_y
            self._f_ff[idx, 10] = f_y

            self._f_ff[idx, 2]  = f_z
            self._f_ff[idx, 5]  = f_z
            self._f_ff[idx, 8]  = f_z
            self._f_ff[idx, 11] = f_z

    #########################
    # Methods for reference #
    #########################
    def _com_ref_in_universe_frame(self):
        # integrate twice m*c_ddot = sum of feedforward forces - m*g
        lo_idx = 0
        td_idx = 0
        for idx, current_time in enumerate(self.time_span[0:-1]):
            f_x = 0.0
            f_y = 0.0
            f_z = 0.0
            for ii in range(0, 4):
                f_x += self._f_ff[idx, 3 * ii]
                f_y += self._f_ff[idx, 3 * ii + 1]
                f_z += self._f_ff[idx, 3 * ii + 2]

            # Acceleration
            if current_time < self.T_stance:
                self.comddot_ref[idx, 0] = 0
                self.comddot_ref[idx, 1] = 0
                self.comddot_ref[idx, 2] = f_z / self._mass #- self._g
            elif self.T_stance <= current_time < self.T_stance + self.T_air:
                self.comddot_ref[idx, 0] = 0.
                self.comddot_ref[idx, 1] = 0.
                self.comddot_ref[idx, 2] = - self._g
            else:
                self.comddot_ref[idx, 0] = 0.
                self.comddot_ref[idx, 1] = 0.
                self.comddot_ref[idx, 2] = 0.


            # Velocity
            if current_time < self.T_stance:
                self.comdot_ref[idx + 1] = self.comdot_ref[idx] + self.comddot_ref[idx] * self._dt
                lo_idx = idx+1
            elif self.T_stance <= current_time < self.T_stance + self.T_air: # ballistic trajectory
                self.comdot_ref[idx+1] = self.comdot_ref[lo_idx] + self.comddot_ref[idx]*current_time
            else:
                self.comdot_ref[idx, 0:3] = 0.

            # Position

            if current_time < self.T_stance:
                self.com_ref[idx + 1] = self.com_ref[idx] + self.comdot_ref[idx] * self._dt
                lo_idx = idx+1
            elif self.T_stance <= current_time < self.T_stance + self.T_air: # ballistic trajectory
                self.com_ref[idx+1] = self.com_ref[lo_idx] + self.comdot_ref[lo_idx]*current_time \
                                    + 0.5*self.comddot_ref[idx]*current_time**2
                td_idx = idx+1
            else:
                self.com_ref[idx, 0:3] = self.com_ref[td_idx, 0:3]

    # Translate CoM trajectory expressed in universe frame [U] to feet trajectory expressed in base frame [B] (useful
    # during stance phase)
    def _feet_ref_in_base_frame(self):
        n_stance = int(np.ceil(self.T_stance/self._dt) + 1) # number of samples in which the robot is in contact with the terrain
        feet_pos_in_universe = self.feet_kinematics(q=self._q_home)
        feet_pos_in_base = self.feet_kinematics(q=self._q_home, reference_frame='base_link')
        for i in range(0, self.samples):
            if i <= n_stance:
                # R_BU and omega_B are set such that no rotations are required
                # but the code allows to modify them without changing everything else
                R_BU = np.eye(3)
                omega_B = np.zeros(3)
                c_U = self.com_ref[i]
                cdot_U = self.comdot_ref[i]

                c_B = self._robot.data.com[0]
                S = pin.skew(omega_B)
                M = np.dot(S, R_BU)

                for foot_idx in range(0, self._nfeet):
                    foot_U = feet_pos_in_universe[foot_idx * 3:(foot_idx + 1) * 3]

                    foot_B = np.dot(R_BU.transpose(), (foot_U - c_U)) + c_B
                    r = np.dot(M, c_B - foot_B)
                    footdot_B = np.dot(R_BU.transpose(), -cdot_U + r)

                    self.feet_pos_ref[i, foot_idx] = foot_B
                    self.feet_lin_vel_ref[i, foot_idx] = footdot_B
            else:
                for foot_idx in range(0, self._nfeet):
                    foot_B = feet_pos_in_base[foot_idx * 3:(foot_idx + 1) * 3]
                    self.feet_pos_ref[i, foot_idx] = foot_B

#     def touch_down(self, q, qdot, t):
#         is_in_touch = False
#
#         if t > self.T_stance+self.T_air/2:
#             feet_pos = self.feet_kinematics(q, qdot)
#             if feet_pos[2] <= self._z_touching or feet_pos[5] <= self._z_touching or feet_pos[8] <= self._z_touching \
#                     or feet_pos[11] == self._z_touching:
#                 is_in_touch = True
#         return is_in_touch

# import os
# import numpy as np
# from numba import jit
# import sys
# import time
# import scipy.linalg as linalg
# from pysolo.controllers.jumping.base import BaseCtrl, PrintStyle
#
# # Pinocchio modules
# import pinocchio as pin  # Pinocchio library
#
# #np.set_printoptions(suppress=True, precision=3, linewidth=os.get_terminal_size(0)[0])
#
# class Thruster(BaseCtrl):
#     def __init__(self, robot, x_max=0., y_max=0., z_max=0., alpha=0.9, settiling_time = 0.2,
#                  only_thruster = False, vectors_for_plots = True):
#         '''
#         Args:
#             alpha: force scaling factor [max force actuated = alpha * max force achievable]
#             dt: sampling time
#             only_thruster: True if the user does not want to use the lander
#             vectors_for_plot: True if the user wants to save all the variables for making plots
#         '''
#         super().__init__(robot)
#         # Arguments
#         self._x_max = x_max
#         self._y_max = y_max
#         self._z_max = z_max
#         self._alpha= alpha
#         self._Bezier_coeffs_normalized = np.array([[0.0, 0.8, 1.0, 1.0], [1.0, 1.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0]])
#         self._Bezier_average = 0.5*(self._Bezier_coeffs_normalized[0] + self._Bezier_coeffs_normalized[1]).mean()
#
#         self._vectors_for_plots = vectors_for_plots
#
#         self._jump_height = self._z_max - self._q_home[2]
#
#         # Jacobian (ee: stack of feet)
#         self.J_xyz_stack = np.zeros([3 * self._nfeet, self._nqa])
#
#         # These fields are filled by calling force_shaping()
#         # T_stance: time horizon for the thrusting phase
#         # T_air: time of flight
#         # T_total = T_stance + T_air + 0.5 (0.5 is added for see what happen after the touch-down, can be changed)
#         # samples: number of time samples between 0.0 and T_total of lenght dt
#         # time_span: vector containing all the sample instants
#         self.T_stance = 0.
#         self.T_air = 0.
#         self.T_total = settiling_time # = T_stance + T_air + settiling_time
#         self.samples = 0
#         self.sample_at_lift_off = 0
#         self.sample_at_touch_down = 0
#         self.time_span = np.zeros(0)
#
#         self._compute_times()
#
#         # order: sample, foot_xyz
#         self._f_ff = np.zeros([self.samples,12])
#         self._force_shaping()
#
#         # These fields are filled by calling reference()
#         # order: sample, xyz
#         self.com_ref = np.zeros([self.samples,3])
#         self.comdot_ref = np.zeros([self.samples,3])
#         self.comddot_ref = np.zeros([self.samples,3])
#
#         # order: sample, feet, xyz
#         self.feet_pos_ref = np.zeros([self.samples,4,3])
#         self.feet_lin_vel_ref = np.zeros([self.samples,4,3])
#
#         self._reference()
#
#         # fb gains
#         kp = 100.0
#         kd = 20.0
#         self._Kp = kp * np.eye(3 * self._nfeet)
#         self._Kd = kd * np.eye(3 * self._nfeet)
#
#
#     def _compute_times(self):
#         f_z_max = self._max_force_along_z()
#
#         self.T_air = np.sqrt(8 * self._jump_height / self._g)
#         self.T_stance = (self._weight * self.T_air) / (4 * f_z_max * self._Bezier_average - self._weight)
#
#
#         self.sample_at_lift_off = int(np.ceil(self.T_stance / self._dt) + 1)
#         self.sample_at_touch_down = self.sample_at_lift_off + int(np.ceil(self.T_air / self._dt) + 1)
#         self.T_total += self.T_stance + self.T_air
#         self.samples = int(np.ceil(self.T_total / self._dt) + 1)
#         self.time_span = np.linspace(0.0, self.T_total, self.samples)
#
#
#     def _force_shaping(self):
#         # Feedforward force is a Bèzier 3rd order polynomial
#         # f_ff(t) = b_0*(1-t)^3 + b_1*t*(1-t)^2 + b_2*t^2*(1-t) + b_3*t^3
#         #         = Bezier_coeffs' * time_vec3(t)
#         time_vec3 = lambda t: np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])
#
#         f_z_max = self._max_force_along_z()
#         f_x_max = self._max_force_along_xy(self._x_max)
#         f_y_max = self._max_force_along_xy(self._y_max)
#
#         Bezier_coeffs_x = f_x_max * self._Bezier_coeffs_normalized
#         Bezier_coeffs_y = f_y_max * self._Bezier_coeffs_normalized
#         Bezier_coeffs_z = f_z_max * self._Bezier_coeffs_normalized
#
#         self._feedforward_force(Bezier_coeffs_x, Bezier_coeffs_y, Bezier_coeffs_z, time_vec3)
#
#
#     def _reference(self):
#         self.com_ref[0, 0:3] = self._q_home[0:3]
#         self._com_ref_in_universe_frame()
#         self._feet_ref_in_base_frame()
#
#
#     # The output of the controller is the torque as a njoints-dimensional vector and the contact forces
#     def feet_PD_controller(self, J_xyz_stack, base_orientation, feet_pos, feet_pos_ref,
#                            feet_lin_vel, feet_lin_vel_ref = None):
#         # stabilizes the feet position about the initial position moving with a frame centered on the CoM with axes parallel to the world frame
#         Kp = self._Kp
#         Kd = self._Kd
#
#         if feet_lin_vel_ref is None:
#             feet_lin_vel_ref = np.zeros_like(feet_lin_vel)
#         err = feet_pos - feet_pos_ref
#         errdot = feet_lin_vel - feet_lin_vel_ref
#
#         # Compute PD feedback
#         w = float(base_orientation[3])
#         x = float(base_orientation[0])
#         y = float(base_orientation[1])
#         z = float(base_orientation[2])
#         R_BU = pin.Quaternion(w, x, y, z).toRotationMatrix()
#         R_BU_stack = linalg.block_diag(R_BU, R_BU, R_BU, R_BU)
#
#         f_fb_B = np.matmul(Kp, err) + np.matmul(Kd, errdot)
#         f_fb = np.matmul(R_BU_stack, f_fb_B).T
#         tau_fb = -np.matmul(J_xyz_stack.T, f_fb).flatten()
#
#         return tau_fb, f_fb
#
#     def joint_PD_controller(self, q, qdot):
#         err = self._q_home[7:] - q
#         Kp = 49. * np.eye(12)
#         Kd = 14.* np.eye(12)
#
#         f_fb = np.zeros(12)
#         tau_fb = - np.dot(Kp, err) - np.dot(Kd, qdot)
#         return tau_fb.flatten(), f_fb
#
#
#
#     def compute(self, q, qdot, sample, current_time):
#         # Execute the thrusting phase
#
#         # Compute/update all the joints and frames
#         self.forward_kinematics(q.copy(), qdot.copy(), 'base_link')
#
#         # Get force
#         f_ff = self._f_ff[sample]
#
#         # Get Jacobians
#         for idx, foot_frame in enumerate(self._feet_frames):
#             J_foot = pin.getFrameJacobian(self._robot.model, self._robot.data, foot_frame,
#                                           self._local_world_aligned_frame)
#             self.J_xyz_stack[idx * 3:(idx + 1) * 3] = J_foot[0:3, 6:].copy()
#
#         # Compute joint torque
#         tau_ff = -np.matmul(self.J_xyz_stack.T, f_ff).flatten()
#
#         tau_fb = np.zeros_like(tau_ff)
#
#         # Compute feet position and velocity
#         # here the vectors are stacked per columns
#         # TODO: are the feet position/velocity correctly read?
#         feet_pos_stack = np.zeros(12)
#         feet_lin_vel_stack = np.zeros(12)
#
#         for idx, frame_id in enumerate(self._feet_frames):
#             feet_pos_stack[idx * 3:(idx + 1) * 3] = self._robot.data.oMf[frame_id].translation.transpose()  # wrt world frame
#             frame = self._robot.model.frames[frame_id]
#             foot_rot = self._robot.data.oMf[frame_id].rotation
#             oMff = pin.SE3(foot_rot, np.zeros(3))
#             iMf = frame.placement
#             f_v = iMf.actInv(self._robot.data.v[frame.parent])
#             feet_lin_vel_stack[idx * 3:(idx + 1) * 3] = oMff.act(f_v).linear.transpose()
#
#         # CONTROL LAW
#         # Track feet reference expressed in base frame (retrieved from com trajectory expressed in universal frame)
#         # Get the reference position and linear velocity of the feet in the base frame
#         p_ref = self.feet_pos_ref[sample]
#         lv_ref = self.feet_lin_vel_ref[sample]
#
#         feet_pos_ref_stack = np.hstack([p_ref[0], p_ref[1], p_ref[2], p_ref[3]])
#         feet_lin_vel_ref_stack = np.hstack([lv_ref[0], lv_ref[1], lv_ref[2], lv_ref[3]])
#
#         tau_fb, f_fb = self.feet_PD_controller(self.J_xyz_stack, q[3:7], feet_pos_stack, feet_pos_ref_stack,
#                                                feet_lin_vel_stack, feet_lin_vel_ref_stack)
#
#         # if current_time < self.T_stance:
#         #     tau_fb, f_fb = self.feet_PD_controller(self.J_xyz_stack, q[3:7], feet_pos_stack, feet_pos_ref_stack, feet_lin_vel_stack, feet_lin_vel_ref_stack)
#         # else:
#         #     tau_fb, f_fb = self.joint_PD_controller(q[7:], qdot[6:])
#
#         tau = tau_ff + tau_fb
#         # check saturations
#         tau_sat = np.clip(tau, self._tau_min, self._tau_max)
#         if False in (tau == tau_sat):
#             print('Thruster:')
#             print(PrintStyle.WARNING + "torque saturation" + PrintStyle.ENDC)
#             print('tau', tau, '\nerr', tau-tau_sat)
#
#         if self._vectors_for_plots:# and sample != 0:
#             data = self._store_all_data(q=q, qdot=qdot, com_ref=self.com_ref[sample], comdot_ref=self.comdot_ref[sample], f=f_ff+f_fb,
#                                         f_ff=f_ff, f_fb=f_fb, tau=tau_sat, tau_fb=tau_fb, tau_ff=tau_ff)
#         else:
#             data = {'torque': tau_sat}
#         return data
#
#
#     #############################
#     # Methods for force shaping #
#     #############################
#
#     def _max_force_along_z(self):
#         # f_max is the force exerted when the robot is in the init config
#         lower_leg_disp = self._robot.model.frames[self._feet_frames[0]].placement.translation
#         lower_leg_length = np.sqrt(np.matmul(lower_leg_disp.T, lower_leg_disp))
#
#         q_FL_hip = self._q_home[8]
#         q_FL_knee = self._q_home[9]
#         f_z_max =  self._alpha * self._tau_max[0] / lower_leg_length * abs(np.sin(q_FL_hip - q_FL_knee))
#
#         return f_z_max
#
#
#     def _max_force_along_xy(self, displacement):
#         # f_x_max is computed approximating the the lift off x-position of the CoM with the initial position of the CoM
#         f_max = displacement * self._mass / (16 * self._Bezier_average * self.T_stance**2)
#         print(displacement, self._mass, self._Bezier_average, self.T_stance**2)
#
#         return f_max
#
#
#     def _feedforward_force(self, Bezier_coeffs_x, Bezier_coeffs_y, Bezier_coeffs_z, time_vec3):
#         # assign to all the feet the same force (f_x, 0, f_z)
#         # return a matrix in which the i-th columns contain the force applied by the contact points of the robot on the
#         # environment at sampling time i
#         f_x = 0.0
#         f_y = 0.0
#         f_z = 0.0
#
#         for idx, current_time in enumerate(self.time_span):
#             if current_time < self.T_stance / 2:
#                 # f_z continuously increases
#                 t = current_time / self.T_stance
#                 f_x = np.dot(Bezier_coeffs_x[0], time_vec3(t))
#                 f_y = np.dot(Bezier_coeffs_y[0], time_vec3(t))
#                 f_z = np.dot(Bezier_coeffs_z[0], time_vec3(t))
#
#
#             elif self.T_stance / 2 <= current_time < self.T_stance:
#                 # f_z continuously decreases
#                 t = 2 * current_time / self.T_stance - 1
#                 f_x = np.dot(Bezier_coeffs_x[1], time_vec3(t))
#                 f_y = np.dot(Bezier_coeffs_y[1], time_vec3(t))
#                 f_z = np.dot(Bezier_coeffs_z[1], time_vec3(t))
#             '''
#             elif self.T_stance <= current_time < self.T_air + self.T_stance:
#                 f_x = 0.0
#                 f_z = 0.0
#
#             elif current_time >= self.T_air + self.T_stance:
#                 f_x = 0.0
#                 #f_z = self._weight/4
#                 f_z = 0.0
#             '''
#             self._f_ff[idx, 0]  = f_x
#             self._f_ff[idx, 3]  = f_x
#             self._f_ff[idx, 6]  = f_x
#             self._f_ff[idx, 9]  = f_x
#
#             self._f_ff[idx, 1]  = f_y
#             self._f_ff[idx, 4]  = f_y
#             self._f_ff[idx, 7]  = f_y
#             self._f_ff[idx, 10] = f_y
#
#             self._f_ff[idx, 2]  = f_z
#             self._f_ff[idx, 5]  = f_z
#             self._f_ff[idx, 8]  = f_z
#             self._f_ff[idx, 11] = f_z
#
#     #########################
#     # Methods for reference #
#     #########################
#     def _com_ref_in_universe_frame(self):
#         # integrate twice m*c_ddot = sum of feedforward forces - m*g
#         for idx, current_time in enumerate(self.time_span[0:-1]):
#             f_x = 0.0
#             f_y = 0.0
#             f_z = 0.0
#             for ii in range(0, 4):
#                 f_x += self._f_ff[idx, 3 * ii]
#                 f_y += self._f_ff[idx, 3 * ii + 1]
#                 f_z += self._f_ff[idx, 3 * ii + 2]
#
#             # Acceleration
#             if current_time < self.T_stance + self.T_air:
#                 self.comddot_ref[idx, 0] = f_x / self._mass
#                 self.comddot_ref[idx, 1] = f_y / self._mass
#                 self.comddot_ref[idx, 2] = f_z / self._mass - self._g
#             else:
#                 self.comddot_ref[idx, 0] = f_x / self._mass
#                 self.comddot_ref[idx, 1] = f_y / self._mass
#                 self.comddot_ref[idx, 2] = f_z / self._mass # at touch-down we have an abrupt change in the z COM trajectory
#
#             # Velocity
#             if current_time < self.T_stance + self.T_air:
#                 self.comdot_ref[idx + 1] = self.comdot_ref[idx] + self.comddot_ref[idx] * self._dt
#             else:
#                 #self.comdot_ref[idx + 1, 0:2] = self.comdot_ref[idx, 0:2] + self.comddot_ref[idx, 0:2] * self._dt
#                 self.comdot_ref[idx + 1, 0:3] = 0.
#
#             # Position
#             if current_time < self.T_stance + self.T_air:
#                 self.com_ref[idx + 1] = self.com_ref[idx] + self.comdot_ref[idx] * self._dt
#             else:
#                 #self.com_ref[idx + 1, 0:2] = self.com_ref[idx, 0:2] + self.comdot_ref[idx, 0:2] * self._dt + \
#                 #                        0.5 * self.comddot_ref[idx, 0:2] * self._dt ** 2
#                 self.com_ref[idx + 1, 0:3] = self.com_ref[idx, 0:3]
#
#
#     # Translate CoM trajectory expressed in universe frame [U] to feet trajectory expressed in base frame [B] (useful
#     # during stance phase)
#     def _feet_ref_in_base_frame(self):
#         n_stance = int(np.ceil(self.T_stance/self._dt) + 1) # number of samples in which the robot is in contact with the terrain
#         feet_pos_in_universe = self.feet_kinematics(q=self._q_home)
#         feet_pos_in_base = self.feet_kinematics(q=self._q_home, reference_frame='base_link')
#         for i in range(0, self.samples):
#             if i <= n_stance:
#                 # R_BU and omega_B are set such that no rotations are required
#                 # but the code allows to modify them without changing everything else
#                 R_BU = np.eye(3)
#                 omega_B = np.zeros(3)
#                 c_U = self.com_ref[i]
#                 cdot_U = self.comdot_ref[i]
#
#                 c_B = self._robot.data.com[0]
#                 S = pin.skew(omega_B)
#                 M = np.dot(S, R_BU)
#
#                 for foot_idx in range(0, self._nfeet):
#                     foot_U = feet_pos_in_universe[foot_idx * 3:(foot_idx + 1) * 3]
#
#                     foot_B = np.dot(R_BU.transpose(), (foot_U - c_U)) + c_B
#                     r = np.dot(M, c_B - foot_B)
#                     footdot_B = np.dot(R_BU.transpose(), -cdot_U + r)
#
#                     self.feet_pos_ref[i, foot_idx] = foot_B
#                     self.feet_lin_vel_ref[i, foot_idx] = footdot_B
#             else:
#                 for foot_idx in range(0, self._nfeet):
#                     foot_B = feet_pos_in_base[foot_idx * 3:(foot_idx + 1) * 3]
#                     self.feet_pos_ref[i, foot_idx] = foot_B
#
#

    def touch_down(self, q, qdot, t):
        is_in_touch = False

        feet_pos = self.feet_kinematics(q, qdot)
        if feet_pos[2] <= self._z_touching or feet_pos[5] <= self._z_touching or feet_pos[8] <= self._z_touching \
                or feet_pos[11] <= self._z_touching:
            is_in_touch = True
        return is_in_touch

    def lift_off(self, q, qdot, t):
        has_lift_off = False

        feet_pos = self.feet_kinematics(q, qdot)
        #print('feet_z', feet_pos[2], feet_pos[5], feet_pos[8], feet_pos[11])
        if feet_pos[2] > self._z_touching and feet_pos[5] > self._z_touching and feet_pos[8] > self._z_touching \
                and feet_pos[11] > self._z_touching:
            has_lift_off = True

        return has_lift_off
