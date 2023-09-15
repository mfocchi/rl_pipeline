import numpy as np
import pinocchio as pin

zeros = np.zeros(19)

class BaseCtrl:
    def __init__(self, robot, tau_max=15, dt=0.001, total_samples = 0):
        '''
        Args:
            robot: Pinocchio model
            tau_max: maximum achievable torque [tau_max = max current * torque constant] //TODO: to be checked
            dt: sampling time
        '''

        self._robot = robot
        self._q_home = robot.q0
        pin.centerOfMass(self._robot.model, self._robot.data, self._q_home)
        self._com_home = self._robot.data.com[0]
        self._dt = dt

        # USEFUL ROBOT INFO
        # Number of variables
        self._njoints = self._robot.model.njoints
        self._nqa = 12  # number of actuated joints
        self._nqu = 7  # number of underactuated joints (3 for postion + 4 for quaternion)
        self._nfeet = 4

        # Actuation limits
        self._tau_max = tau_max * np.ones([self._nqa])
        self._tau_min = -self._tau_max

        # Compute robot total mass and weight
        self._mass = pin.computeTotalMass(self._robot.model, self._robot.data)
        self._g = abs(self._robot.model.gravity.linear[2])  # 9.81
        self._weight = self._mass * self._g

        # Frames: feet and universe
        self._feet_names = []

        for frame in self._robot.model.frames:
            if 'foot' in frame.name:
                self._feet_names.append(frame.name)

        self._feet_frames = [self._robot.model.getFrameId(fname) for fname in self._feet_names]
        self._local_world_aligned_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED  # frame centered on the moving part but with axes aligned with the WORLD frame

        self._z_touching = 0.018 # vertical distance of a foot frame from the contact surface when the foot is in contact



    def forward_kinematics(self, q, qdot=None, reference_frame='universe'):
        model = self._robot.model
        data = self._robot.data
        if qdot is None:
            qdot = np.zeros(len(q) - 1)
        if reference_frame == 'base_link':
            q[:7] = [0., 0., 0., 0., 0., 0., 1.]
            qdot[:6] = [0., 0., 0., 0., 0., 0.]
        pin.forwardKinematics(model, data, q, qdot)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)


    def feet_kinematics(self, q, qdot=None, reference_frame='universe'):
        # Compute feet pose for the given config in the specified reference frame
        if qdot is None:
            qdot = np.zeros(len(q) - 1)

        self.forward_kinematics(q.copy(), qdot.copy(), reference_frame)
        feet_pos = np.zeros(12)
        for idx, frame_id in enumerate(self._feet_frames):
            feet_pos[idx * 3:(idx + 1) * 3] = self._robot.data.oMf[frame_id].translation.transpose().copy()

        return feet_pos

    def forward_dynamics(self, model, data, q, qdot, tau, fc = None):
        # q: base position and orientation + joint configuration    [size: model.nq]
        # qdot : base linear and angular velocity + joint velocity  [size: model.nv]
        # tau: actuation torque                                     [size: model.na]
        # fc: vertical stack of contact forces [LF, RF, LH, RH]     [size: 12]
        tau = np.concatenate([zeros[0:6], tau], axis=0)
        if fc is None:
            pin.aba(model, data, q, qdot, tau)
        else:
            fext_local_frame = []
            offset = 0
            joint_refs = [4, 7, 10, 13]
            for i in range(2, 14):
                # if a link is in contact with the ground
                if i in joint_refs:
                    R = data.oMi[i].rotation                 # orientation of the KFE joint frame wrt universe frame
                    r = data.liMi[13].translation   # distance vector beetween feet frame and KFE joint frame (parent)

                    F_local = np.dot(R, fc[3*offset: 3*(offset+1)])
                    M_local = np.dot(pin.skew(r), F_local)
                # otherwise
                else:
                    F_local = np.zeros(3)
                    M_local = np.zeros(3)

                F = pin.Force(np.concatenate([F_local, M_local]))
                fext_local_frame.append(F)

            qddot = pin.aba(model, data, q, qdot, tau, fext_local_frame)

        return qddot

    def integrate(self, q, qdot, qddot):
        qdot_mean = qdot + 0.5 * self._dt * qddot
        qdot += self._dt * qddot
        q = pin.integrate(self._robot.model, q, qdot_mean * self._dt)

        return q, qdot




    def _store_all_data(self, q=zeros, qdot=zeros[0:18], com=zeros[0:3], comdot=zeros[0:3], com_ref=zeros[0:3], comdot_ref=zeros[0:3], f=zeros[0:12],
                       f_ff=zeros[0:12], f_fb=zeros[0:12], tau=zeros[0:12], tau_fb=zeros[0:12], tau_ff=zeros[0:12]):

        pin.centerOfMass(self._robot.model, self._robot.data, q, qdot)

        com = self._robot.data.com[0]
        comdot = self._robot.data.vcom[0]
        force = f
        force_fb = f_fb
        force_ff = f_ff
        torque = tau
        torque_fb = tau_fb
        torque_ff = tau_ff
        data = {'q': q, 'qdot': qdot, 'com': com, 'comdot':comdot,'com_ref': com_ref, 'comdot_ref': comdot_ref, 'force': force,
                'force_fb': force_fb, 'force_ff': force_ff, 'torque': torque, 'torque_fb': torque_fb,
                'torque_ff': torque_ff}

        return data


    def compute(self, q, qdot, sample, current_time):
    # this method must be override
        tau = np.zeros(self._nqa)
        return tau



class PrintStyle:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'