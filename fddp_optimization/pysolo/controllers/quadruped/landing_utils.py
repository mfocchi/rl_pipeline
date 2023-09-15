import numpy as np
import scipy.linalg as LA

import pinocchio as pin
import tsid

class LandingController:
    def __init__(self, robot, timeStep, max_knots):
        self.robot_tsid = tsid.RobotWrapper(robot.model, False)
        self.q0 = robot.q0

        self.timeStep = timeStep
        self.max_knots = max_knots
        self.remaning_knots = max_knots + 1

        # MSD params
        self.m = robot.robot_mass
        self.K = 1000
        self.D = 2*np.sqrt(self.K*self.m)
        self.L = self.q0[2] # L is the desired distance of the COM from the contact surface (assumed flat)

        # OCP params
        # alpha_sq is used to give more importance to a direction instead of the other
        # rho_sq is a trade-off between null velocity and perfect position matching at steady-state
        self.alpha_sq_x = 1
        self.rho_sq_x = 1
        self.alpha_sq_y = 1
        self.rho_sq_y = 1



        self.formulation_tsid = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot_tsid, False)

        t = 0
        q = pin.neutral(self.robot_tsid.model())
        v = pin.utils.zero(self.robot_tsid.nv)

        data = self.formulation_tsid.computeProblemData(0, q, v) # TODO: is it needed?

        self.data_tsid = self.formulation_tsid.data()

        self.robot_tsid.computeAllTerms(self.data_tsid, q, v)

        ############
        # CONTACTS #
        ############
        # TODO: modify here for sloped surfaces (and be consistent with crocoddyl)
        contactNormal = np.array([0., 0., 1.])
        mu = 0.3
        fMin = 0.0
        fMax = 100.0

        kp_contact = 10.
        w_forceRef = 1e-5

        self.contacts = 4*[None]

        feet_frames = []
        for frame in self.robot_tsid.model().frames:
            if 'foot' in frame.name:
                feet_frames.append(frame)

        for i, frame in enumerate(feet_frames):
            fname = frame.name
            fidx = self.robot_tsid.model().getFrameId(fname)
            self.contacts[i] = tsid.ContactPoint(fname, self.robot_tsid, fname, contactNormal, mu, fMin, fMax)
            self.contacts[i].setKp(kp_contact * np.ones(3))
            self.contacts[i].setKd(2.0 * np.sqrt(kp_contact) * np.ones(3))
            self.formulation_tsid.addRigidContact(self.contacts[i], w_forceRef, 1.0, 1)
            H_rf_ref = self.robot_tsid.framePosition(self.data_tsid, fidx)
            self.contacts[i].setReference(H_rf_ref)
            self.contacts[i].useLocalFrame(False)
            self.formulation_tsid.addRigidContact(self.contacts[i], w_forceRef, 1.0, 1)

        ############
        # COM TASK #
        ############
        kp_com = self.K / self.m
        w_com = 10

        self.comTask = tsid.TaskComEquality("task-com", self.robot_tsid)
        self.comTask.setKp(kp_com * np.ones(3))
        self.comTask.setKd(2.0 * np.sqrt(kp_com) * np.ones(3))
        self.formulation_tsid.addMotionTask(self.comTask, w_com, 1, 0.0)

        self.comTraj = tsid.TrajectoryEuclidianConstant("traj-com")
        self.comSample = self.comTraj.computeNext()

        #################
        # POSTURAL TASK #
        #################
        kp_posture = 10.
        w_posture = 1e-5

        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot_tsid)
        self.postureTask.setKp(kp_posture * np.ones(12))
        self.postureTask.setKd(2.0 * np.sqrt(kp_posture) * np.ones(12))
        self.formulation_tsid.addMotionTask(self.postureTask, w_posture, 1, 0.0)

        self.postureTraj = tsid.TrajectoryEuclidianConstant("traj-posture", self.q0[7:])
        self.postureSample = self.postureTraj.computeNext()

        ##########
        # SOLVER #
        ##########

        self.solver_tsid = tsid.SolverHQuadProgFast("qp solver")
        self.solver_tsid.resize(self.formulation_tsid.nVar, self.formulation_tsid.nEq, self.formulation_tsid.nIn)



    def state_propagation_matrices(self, omega_sq, knots):
        Ad = np.zeros([2, 2, knots+1])
        Bd = np.zeros([2, 1, knots+1])

        A = np.zeros([2, 2])
        B = np.zeros([2, 1])

        PHI = np.eye(2)
        GAMMA = np.zeros([2, 1])

        for k in range(1, knots):
            A = np.array([ [    0.0, 1.0],
                           [omega_sq[k], 0.0] ])
            B = np.array([ [     0.0],
                           [-omega_sq[k]] ])

            Ad[:, :, k - 1] = np.eye(2) + self.timeStep * A
            Bd[:, :, k - 1] = self.timeStep * B

            PHI = np.matmul(Ad[:, :, k - 1], PHI)
            GAMMA = np.matmul(Ad[:, :, k - 1], GAMMA) + Bd[:, :, k - 1]

        return Ad, Bd, PHI, GAMMA


    def weight_matrices(self, PHI, GAMMA, alpha_sq=1, rho_sq=1):
        # Selector matrices
        P = np.array([[1, 0]])
        V = np.array([[0, 1]])

        # COST MATRICES
        # cost_i =          state_TD.T*M1*state_TD + u.T*M2*u + 2*state_TD.T*M3*u     # terms coming from (c_dot_i_N)^2
        #           + rho^2*(state_TD.T*N1*state_TD + u.T*n2*u + 2*state_TD.T*N3*u)   # terms coming from (c_i_N - zmp_i)^2
        # with i = x, y

        m1 = np.matmul(V, PHI)
        m2 = np.matmul(V, GAMMA)

        n1 = np.matmul(P, PHI)
        n2 = np.matmul(P, GAMMA)

        M1 = np.matmul(m1.T, m1)
        M2 = np.matmul(m2.T, m2)
        M3 = np.matmul(m1.T, m2)

        N1 = np.matmul(n1.T, n1)
        N2 = np.matmul(n2.T, n2) + 1 - 2* n2.T
        N3 = np.matmul(n1.T, n2) - n1.T

        W1 = alpha_sq * (M1 + rho_sq * N1)
        W2 = alpha_sq * (M2 + rho_sq * N2)
        W3 = alpha_sq * (M3 + rho_sq * N3)

        return W1, W2, W3


    def xy_dynamics(self, state_xy0, omega_sq, Ad, Bd, zmp, knots):
        # x_{k+1} = Ad_k * x_k + B_k * zmp
        # and the same for y

        state = state_xy0

        p_com_ref = np.zeros(knots+1)
        v_com_ref = np.zeros(knots+1)
        a_com_ref = np.zeros(knots+1)


        k = 0
        for k in range(0, knots):

            p_com_ref[k] = state[0]
            v_com_ref[k] = state[1]
            a_com_ref[k] = omega_sq[k] * (state[0] - zmp)

            state = np.dot(Ad[:, :, k], state) + np.dot(Bd[:, :, k], zmp)
        print('p shape = ', p_com_ref.shape)
        print('k = ', k)
        p_com_ref[k] = state[0]
        v_com_ref[k] = state[1]
        a_com_ref[k] = omega_sq[k] * (state[0] - zmp)

        return p_com_ref, v_com_ref, a_com_ref

    def z_dynamics(self, state_z0, knots):
        # Compute reference for com_z using MSD equation assuming the touch down occurring at the current time
        # m*com_z_ddot + D*com_z_dot + K*(com_z-L) = -m*g
        # state:      com_z_dot
        #               com_z-L
        # input: -g
        # output:    com_z_ddot
        #             com_z_dot
        #               com_z-L
        #
        # The initial conditions for the dynamics p_com0 and v_com0 are obtained from the current state of the robot

        # Use state augmentation technique for integrating the dynamics
        # augmented_state: state
        #                  input
        # augmented_state_dot = A_bar * augmented state
        # A_bar = [A, B] #nonzeros
        #         [0, 0] #zeros (in structural sense)

        m = self.m
        D = self.D
        K = self.K
        L = self.L
        g = self.robot_tsid.model().gravity.linear[2]

        A_bar = np.array([[-D/m, -K/m,  1.],
                          [  1.,   0.,  0.],
                          [  0.,   0.,  0.]])

        A_d = LA.expm(A_bar * self.timeStep)

        augmented_state = np.array([[state_z0[1]],
                                    [state_z0[0] - L],
                                    [-g]])

        p_com_z_ref = np.zeros(knots+1)
        v_com_z_ref = np.zeros(knots+1)
        a_com_z_ref = np.zeros(knots+1)

        p_com_z_ref[0] = state_z0[0]
        v_com_z_ref[0] = state_z0[1]
        a_com_z_ref[0] = -g

        omega_sq = np.zeros(knots+1)

        for k in range(1, knots+1):
            augmented_state = np.matmul(A_d, augmented_state)

            p_com_z_ref[k] = augmented_state[1] + L
            v_com_z_ref[k] = augmented_state[0]
            a_com_z_ref[k] = np.matmul(A_bar[0, :], augmented_state)

            omega_sq[k] = (g + a_com_z_ref[k]) / p_com_z_ref[k]
            # omega_sq[0]=0 because a_com_z_ref[0]=-g

        return p_com_z_ref, v_com_z_ref, a_com_z_ref, omega_sq

    def com_ref(self, q, v, knots):
        # compute current COM
        pin.centerOfMass(self.robot_tsid.model(), self.robot_tsid.data(), q, v, False)
        p_com = self.robot_tsid.data().com[0]
        v_com = self.robot_tsid.data().vcom[0]

        # state_x (state_y, state_z) contains position and velocity at current time of the COM in the x (y, z) direction
        state_x = np.array([[p_com[0]],
                            [v_com[0]]])

        state_y = np.array([[p_com[1]],
                            [v_com[1]]])

        state_z = np.array([[p_com[2]],
                            [v_com[2]]])

        # MSD equation
        # omega_sq[k] = (g + a_com_z_ref[k]) / p_com_z_ref[k]
        p_com_z_ref, v_com_z_ref, a_com_z_ref, omega_sq = self.z_dynamics(state_z, knots)

        # COMPUTE ZMP
        # Ad, Bd, PHI and GAMMA are the same for both com_x and com_y because the latter dynamics depend only on omega
        Ad, Bd, PHI, GAMMA = self.state_propagation_matrices(omega_sq, knots)

        W1_x, W2_x, W3_x = self.weight_matrices(PHI, GAMMA, alpha_sq=self.alpha_sq_x, rho_sq=self.rho_sq_x)
        W1_y, W2_y, W3_y = self.weight_matrices(PHI, GAMMA, alpha_sq=self.alpha_sq_y, rho_sq=self.rho_sq_y)

        M_x = - W3_x.T / W2_x
        M_y = - W3_y.T / W2_y

        zmp_x = np.matmul(M_x, state_x)
        zmp_y = np.matmul(M_y, state_y)

        # PROPAGATE xy DYNAMICS
        # Maybe one can compute com x-y reference only the first sample, not all the time horizon
        p_com_x_ref, v_com_x_ref, a_com_x_ref = self.xy_dynamics(state_x, omega_sq, Ad, Bd, zmp_x, knots)
        p_com_y_ref, v_com_y_ref, a_com_y_ref = self.xy_dynamics(state_y, omega_sq, Ad, Bd, zmp_y, knots)

        # Take ONLY the first knot
        p_com_ref = np.array([p_com_x_ref[0], p_com_y_ref[0], p_com_z_ref[0]])
        v_com_ref = np.array([v_com_x_ref[0], v_com_y_ref[0], v_com_z_ref[0]])
        a_com_ref = np.array([a_com_x_ref[0], a_com_y_ref[0], a_com_z_ref[0]])

        return p_com_ref, v_com_ref, a_com_ref




    def feedback(self, x):
        self.remaning_knots -= 1
        knots = self.remaning_knots
        q = x[:self.robot_tsid.nq]
        v = x[self.robot_tsid.nq:]
        p_com_ref, v_com_ref, a_com_ref = self.com_ref(q, v, knots)

        # COM TASK
        self.comSample.pos(p_com_ref)
        self.comSample.vel(v_com_ref)
        self.comSample.acc(a_com_ref)

        self.comTask.setReference(self.comSample)

        # POSTURE TASK
        # TODO: can these lines be delated?
        self.postureSample = self.postureTraj.computeNext()
        self.postureTask.setReference(self.postureSample)

        # Formulate and solve the problem
        HQPData = self.formulation_tsid.computeProblemData(0, q, v)
        sol = self.solver_tsid.solve(HQPData)
        if (sol.status != 0):
            print("Time %.3f QP problem could not be solved! Error code:" % t, sol.status)
            exit(-1)
        tau = self.formulation_tsid.getActuatorForces(sol)

        return p_com_ref, v_com_ref, a_com_ref, tau




class PIDController:
    def __init__(self, kp=None, ki=None, kd=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def feedbackPD(self, q_des, q, qd_des, qd):
        tau_fb = self.kp * np.subtract(q_des, q) + self.kd * (qd_des - qd)
        return tau_fb





