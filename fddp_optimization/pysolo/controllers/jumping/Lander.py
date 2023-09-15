
"""
Assumption:
1. the mass is concentrated in the main body 
2. the variation of angular momentum is zero for all the time
3. all the feet land in the same time instant
4. flat horizontal terrain

1. => the inertia does not depends on the robot configuration
2. => no rotations of the base
2., 3., 4. => the main body, the convex hull made by the feet and the are parallel for all the time



MEASURES
time            [s]
position        [m] 
velocity        [m/s]
acceleration    [m/s^2]
mass            [kg]
"""

import numpy as np
# Pinocchio modules
import pinocchio as pin  # Pinocchio library
import scipy.linalg as LA
import tsid
from pysolo.controllers.utils.base_controller import BaseCtrl


class Lander(BaseCtrl):
    def __init__(self, robot, T_total,
                 rho_sq=0.01, K = 1000.0, vectors_for_plots = True):
        '''
        Args:
            T_stance: for t in [0, T_stance) the robot is in contact with the ground
            T_air: for t in [T_stance, T_stance+T_air) the robot is in flight
            T_total: T_stance + T_air + epsilon for settling
            rho_sq: trade-off between velocity and displacement in OCP
            D: damper [Ns/m] in the MSD equation
            K: spring [N/m] in the MSD equation
            vectors_for_plot: True if the user wants to save all the variables for making plots
        '''
        super().__init__(robot)

        self._com_touch_down = np.array([0, 0, 0])
        self._comdot_touch_down = np.array([0, 0, 0])
        self._rho_sq = rho_sq
        self._damping = 2*np.sqrt(K*self._mass)
        self._spring = K
        self._vectors_for_plots = vectors_for_plots

        #TIME VARIABLES
        # sampling times of landing controller
        # {T_stance+T_air, T_stance+T_air+dt, ..., T_total = T_stance+T_air+T_control = T_stance+T_air+N*dt
        self.T_landing = T_total # -T_touchdown: T_landing is the length of the time interval in which the landing controller is on
        self.samples = 0
        self.lander_sample = -1

        # REFERENCES (to be reshaped and filled)
        self.com_z_ref = np.zeros(self.samples)
        self.com_z_dot_ref = np.zeros(self.samples)
        self.com_z_ddot_ref = np.zeros(self.samples)


        self.com_x_ref = np.zeros(self.samples)
        self.com_x_dot_ref = np.zeros(self.samples)
        self.com_x_ddot_ref = np.zeros(self.samples)

        self.ISE = np.zeros(3)


        self.omega_sq = np.zeros(self.samples)
        self.zmp_x = 0.0

        #######################
        # TSID INITIALIZATION
        #######################
        self._robot_tsid = 0

        self._formulation_tsid = 0

        self._data_tsid = 0


        # CONTACTS
        self._contacts = 4 * [None]


        # COM TASK

        self._comTask = 0
        self._comTraj = 0
        self._comSample = 0

        #
        #
        # # ZMP TASK
        # kp_cop = np.array([100.])
        # w_cop = 1e-1
        #
        # self._copTask = tsid.TaskCopEquality("task-cop", self._robot_tsid)
        #
        # self._formulation_tsid.addForceTask(self._copTask, w_cop, 1, 0.0)


        # POSTURAL TASK
        self._postureTask = 0
        self._postureTraj = 0
        self._postureSample = 0

        self._solver_tsid = 0



    def _tsid_start(self, q, qdot, t):

        self._robot_tsid = tsid.RobotWrapper(self._robot.urdf_filename,
                                             ['/home/froscia/solo/pysolo/pysolo/models/robot_description/urdf/meshes'],
                                             pin.JointModelFreeFlyer(), False)

        self._formulation_tsid = tsid.InverseDynamicsFormulationAccForce("tsid", self._robot_tsid, False)

        self._formulation_tsid.computeProblemData(t, q, qdot)
        self._data_tsid = self._formulation_tsid.data()

        self._robot_tsid.computeAllTerms(self._data_tsid, q, qdot)

        ############
        # CONTACTS #
        ############
        contactNormal = np.array([0., 0., 1.])
        mu = 0.3
        fMin = 0.0
        fMax = 100.0

        kp_contact = 10.
        w_forceRef = 1e-5

        for i, fname in enumerate(self._feet_names):
            self._contacts[i] = tsid.ContactPoint(fname, self._robot_tsid, fname, contactNormal, mu, fMin, fMax)
            self._contacts[i].setKp(kp_contact * np.ones(3))
            self._contacts[i].setKd(2.0 * np.sqrt(kp_contact) * np.ones(3))
            self._formulation_tsid.addRigidContact(self._contacts[i], w_forceRef, 1.0, 1)
            H_rf_ref = self._robot_tsid.framePosition(self._data_tsid, self._feet_frames[i])
            self._contacts[i].setReference(H_rf_ref)
            self._contacts[i].useLocalFrame(False)
            self._formulation_tsid.addRigidContact(self._contacts[i], w_forceRef, 1.0, 1)

        ############
        # COM TASK #
        ############
        kp_com = self._spring/self._mass
        w_com = 10

        self._comTask = tsid.TaskComEquality("task-com_z", self._robot_tsid)
        self._comTask.setMask(np.array([1, 0, 1]))  # control only the z coordinate
        self._comTask.setKp(kp_com * np.ones(3))
        self._comTask.setKd(2.0 * np.sqrt(kp_com) * np.ones(3))
        self._formulation_tsid.addMotionTask(self._comTask, w_com, 1, 0.0)

        self._comTraj = tsid.TrajectoryEuclidianConstant("traj-com")
        self._comSample = self._comTraj.computeNext()

        #################
        # POSTURAL TASK #
        #################
        kp_posture = 10.
        w_posture = 1e-5

        self._postureTask = tsid.TaskJointPosture("task-posture", self._robot_tsid)
        self._postureTask.setKp(kp_posture * np.ones(12))
        self._postureTask.setKd(2.0 * np.sqrt(kp_posture) * np.ones(12))
        self._formulation_tsid.addMotionTask(self._postureTask, w_posture, 1, 0.0)

        self._postureTraj = tsid.TrajectoryEuclidianConstant("traj-posture", self._q_home[7:])
        self._postureSample = self._postureTraj.computeNext()

        ##########
        # SOLVER #
        ##########

        self._solver_tsid = tsid.SolverHQuadProgFast("qp solver")
        self._solver_tsid.resize(self._formulation_tsid.nVar, self._formulation_tsid.nEq, self._formulation_tsid.nIn)




    def touch_down(self, q, qdot, t):

        self.T_landing -= t
        self.sample_at_touch_down = int(t/self._dt)
        self.samples = int(self.T_landing/self._dt)
        print(20*'*', 'samples for landing', self.samples)
        print('T_landing', self.T_landing)
        print('t',t)

        # REFERENCES (to be filled)
        self.com_z_ref = np.zeros(self.samples)
        self.com_z_dot_ref = np.zeros(self.samples)
        self.com_z_ddot_ref = np.zeros(self.samples)

        self.com_x_ref = np.zeros(self.samples)
        self.com_x_dot_ref = np.zeros(self.samples)
        self.com_x_ddot_ref = np.zeros(self.samples)

        self.omega_sq = np.zeros(self.samples)

        pin.centerOfMass(self._robot.model, self._robot.data, q, qdot)
        self._com_touch_down = self._robot.data.com[0]
        self._comdot_touch_down = self._robot.data.vcom[0]

        self.com_z_ref[0] = self._com_touch_down[2]
        self.com_z_dot_ref[0] = self._comdot_touch_down[2]
        self.com_z_ddot_ref[0] = -self._g

        self.com_x_ref[0] = self._com_touch_down[0]
        self.com_x_dot_ref[0] = self._comdot_touch_down[0]

        self._z_dynamics()
        self._compute_zmp()

        self._tsid_start(q, qdot, t)


    def _z_dynamics(self):
        # MSD equation to compute com_z dynamics
        # m*com_z_ddot + D*com_z_dot + K*(com_z_0-com_z) = -m*g
        # state:     com_z_dot
        #        com_z_0-com_z
        # input: -g
        # output:    com_z_ddot
        #             com_z_dot
        #         com_z_0-com_z

        m = self._mass
        D = self._damping
        K = self._spring
        L = self.com_z_ref[0]
        print('L', L)

        # Continuous-time model
        A = np.array([[-D/m, -K/m],
                      [  1.,   0.]])
        B = np.array([[1.],
                      [0.]])

        # Use state augmentation technique for integrating the dynamics
        # augmented_state: state
        #                  input
        # augmented_state_dot = A_bar * augmented state
        # A_bar = [A, B] #nonzeros
        #         [0, 0] #zeros (in structural sense)
        nonzeros = np.hstack([A, B])
        zeros = np.zeros([1, 3])
        A_bar = np.vstack([nonzeros,
                           zeros])

        A_d = LA.expm(A_bar * self._dt)

        augmented_state = np.array([[self.com_z_dot_ref[0]],
                                      [self.com_z_ref[0]-L],
                                                [-self._g]])

        for k in range(0, self.samples):
            augmented_state = np.matmul(A_d, augmented_state)
            self.com_z_ddot_ref[k] =  np.matmul(A_bar[0, :], augmented_state)
            self.com_z_dot_ref[k] = augmented_state[0]
            self.com_z_ref[k] = augmented_state[1]+L
            self.omega_sq[k] = (self._g + self.com_z_ddot_ref[k]) / self.com_z_ref[k]
            # omega_sq[0]=0 because self.com_z_ddot_ref[0]=-g


    def _compute_zmp(self):
        # time-vaying dynamics: A and its integrals from 0 to dt does not commutes, then a closed form solution does not
        # exist. Use EULER integration
        # state:     com_x
        #        com_x_dot
        state = np.array([[self._com_touch_down[0]],
                          [self._comdot_touch_down[0]]])
        print('com(TD) = ',self._com_touch_down)
        print('com_dot(TD) = ', self._comdot_touch_down)
        #exit()

        A_d = np.zeros([2, 2, self.samples])
        B_d = np.zeros([2, 1, self.samples])

        A = np.zeros([2,2])
        B = np.zeros([2,1])


        PHI = np.eye(2)
        GAMMA = np.zeros([2,1])

        for k in range(1, self.samples):
            A = np.array([[             0.0,   1.0],
                          [self.omega_sq[k-1], 0.0]])
            B = np.array([               [0.0],
                          [-self.omega_sq[k-1]]])

            A_d[:, :, k-1] = np.eye(2)+self._dt*A
            B_d[:, :, k-1] = self._dt*B

            PHI = np.matmul(A_d[:, :, k - 1], PHI)
            GAMMA = np.matmul(A_d[:, :, k - 1], GAMMA) + B_d[:, :, k - 1]

        # cost = c_dot_0_N^2 + rho^2(c_x_N - zmp_x)^2
        # it can be rewritten as a quadratic function of x_0 (= c_x_0, c_dot_x_0) and u (= zmp_x)

        P = np.array([[1, 0]])
        V = np.array([[0, 1]])

        m1 = np.matmul(V, PHI)
        m2 = np.matmul(V, GAMMA)

        n1 = np.matmul(P, PHI)
        n2 = np.matmul(P, GAMMA)


        M1 = np.matmul(m1.transpose(), m1)
        M2 = np.matmul(m2.transpose(), m2)
        M3 = np.matmul(m1.transpose(), m2)

        N1 = np.matmul(n1.transpose(), n1)
        N2 = np.matmul(n2.transpose(), n2) + np.eye(GAMMA.shape[1]) -2 * n2.transpose()
        N3 = np.matmul(n1.transpose(), n2) - n1.transpose()

        num_zmp = (M3 + self._rho_sq * N3).transpose()
        den_zmp = (M2 + self._rho_sq * N2).transpose()
        coeff_zmp = - num_zmp / den_zmp
        zmp = np.matmul(coeff_zmp, state)

        # Propagate the dynamics
        for k in range(0, self.samples-1):
            state = np.dot(A_d[:, :, k], state) + np.dot(B_d[:, :, k], zmp)
            self.com_x_ref[k+1] = state[0]
            self.com_x_dot_ref[k+1] = state[1]
            self.com_x_ddot_ref[k+1] = self.omega_sq[k]*(state[0] - zmp)

        self.zmp_x = zmp




    def compute(self, q, qdot, sample, t):
        # TODO: for a solution in MPC fashion, add _compute_zmp(sample) here

        # compute torque


        com_ref = np.array([self.com_x_ref[self.lander_sample], 0., self.com_z_ref[self.lander_sample]])
        comdot_ref = np.array([self.com_x_dot_ref[self.lander_sample], 0., self.com_z_dot_ref[self.lander_sample]])
        comddot_ref = np.array([self.com_x_ddot_ref[self.lander_sample], 0., self.com_z_ddot_ref[self.lander_sample]])

        self._comSample.pos(com_ref)
        self._comSample.vel(comdot_ref)
        self._comSample.acc(comddot_ref)
        self._comTask.setReference(self._comSample)


        self._postureSample = self._postureTraj.computeNext()
        self._postureTask.setReference(self._postureSample)

        #self._solver_tsid.resize(self._formulation_tsid.nVar, self._formulation_tsid.nEq, self._formulation_tsid.nIn)
        HQPData = self._formulation_tsid.computeProblemData(t, q, qdot)

        sol = self._solver_tsid.solve(HQPData)
        if (sol.status != 0):
            print("Time %.3f QP problem could not be solved! Error code:" % t, sol.status)
            exit(-1)
        tau = self._formulation_tsid.getActuatorForces(sol)
        qddot = self._formulation_tsid.getAccelerations(sol)


        # print(10*'*', 'Iteration #', sample, 10*'*')
        # print('com_task_ref', self._comTask.position_ref)
        # print('com_ref', com_ref)
        # print('com', self._comTask.position)
        # print('posture_err', q[7:]-self._postureTask.position_ref)
        # print('\ntau', tau)
        self.ISE += self._dt*(self._comTask.position_ref-self._comTask.position)**2


        if self._vectors_for_plots:  # and sample != 0:
            data = self._store_all_data(q=q, qdot=qdot, com_ref=com_ref, comdot_ref=comdot_ref, tau = tau)
        else:
            data = {'torque': tau}

        #data = self._store_all_data()

        self.lander_sample += 1

        #q, qdot = self.integrate(q, qdot, qddot)

        return data#, q, qdot, qddot
