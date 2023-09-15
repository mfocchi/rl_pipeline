import pinocchio as pin
import crocoddyl
import numpy as np
from pysolo.controllers.ddp_utils import config_solo_ddp as conf
import rospy

def ddp_problem(x0, gait, key, value):
    if key == 'walking':
        # Creating a walking problem
        ddp_prob = crocoddyl.SolverBoxDDP(
            gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                      value['stepKnots'], value['supportKnots']))
    elif key == 'jumping':
        # Creating a jumping problem (variations on xyz allowed)
        ddp_prob = crocoddyl.SolverFDDP(
            gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                      value['groundKnots'], value['flyingKnots']))
    elif key == 'jumping_limited':
        # Creating a jumping problem (variations on xyz allowed)
        ddp_prob = crocoddyl.SolverFDDP(
            gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                      value['groundKnots'], value['flyingKnots']))
    elif key == 'jumping+yaw':
        # Creating a jumping problem (variations on xyz+Y allowed)
        timeStep = value['timeStep']
        groundKnots = int(value['launchingT'] / timeStep)
        flyingKnots = int(value['flyingUpT'] / timeStep)
        ddp_prob = crocoddyl.SolverFDDP(
            gait.createJumpingYawProblem(x0, value['jumpHeight'], value['jumpLength'], value['yaw'], timeStep,
                                         groundKnots, flyingKnots))

    elif key == 'somersault':
        # Creating a somersault problem (variations on xyz+P allowed)
        ddp_prob = crocoddyl.SolverDDP(
            gait.createSomersaultProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                         value['groundKnots'], value['flyingKnots']))

    elif key == 'wobbling':
        # Creating a somersault problem (variations on RPY allowed)
        ddp_prob = crocoddyl.SolverFDDP(
            gait.createBaseWobblingProblem(x0, value['timeStep'], value['frequency'], value['amplitudes'],
                                           value['totalKnots']))

    elif key == 'jumping+slope':
        # Creating a jumping problem on inclined plane
        ddp_prob = crocoddyl.SolverBoxDDP(
            gait.createJumpingOnSlopedPlane(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                            value['groundKnots'], value['flyingKnots'], value['RotPlane0'],
                                            value['RotPlane1'])
        )

    elif key == 'jump_froscia':
        timeStep = value['timeStep']
        groundKnots = int(value['launchingT'] / timeStep)
        flyingKnots = int(value['flyingUpT'] / timeStep)
        deg2rad = np.pi/180
        Rstart_list = [None]*4
        for i, rpy in enumerate(value['rpy_feet_start']):
            r_rad = rpy[0] * deg2rad
            p_rad = rpy[1] * deg2rad
            y_rad = rpy[2] * deg2rad
            Rstart_list[i] = pin.rpy.rpyToMatrix(r_rad, p_rad, y_rad)

        Rend_list = [None] * 4
        for i, rpy in enumerate(value['rpy_feet_end']):
            r_rad = rpy[0] * deg2rad
            p_rad = rpy[1] * deg2rad
            y_rad = rpy[2] * deg2rad
            Rend_list[i] = pin.rpy.rpyToMatrix(r_rad, p_rad, y_rad)

        ddp_prob = crocoddyl.SolverFDDP(
            gait.createJumpingOnSlopedPlane(x0, value['jumpHeight'], value['jumpLength'], timeStep, groundKnots,
                                            flyingKnots, Rstart_list, Rend_list)
        )

    return ddp_prob



def set_ddp_callbacks(robot, ddp_i, useGazebo):
    if useGazebo:
        ddp_i.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    else:
        cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        disp = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, frameNames=['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot'])
        ddp_i.setCallbacks(
            [crocoddyl.CallbackLogger(),
             crocoddyl.CallbackVerbose(),
             crocoddyl.CallbackDisplay(disp)])



def printTask(gait_phases, i, verbose=True):
    print('\nSimulating task', i, ':', list(gait_phases[i].keys())[0])
    if verbose:
        task_keys = list(gait_phases[i].values())[0].keys()
        for k in task_keys:
            print('\t', k, ':', list(gait_phases[i].values())[0].get(k))


class DDPInterpolate:
    def __init__(self, p, ddp, ddp_dt, sim_dt, start_q=None, start_qd = None, savefile=None):
        self.p = p
        self.ddp = ddp
        self.alpha = sim_dt / ddp_dt
        self.window_size = int(1/self.alpha)
        self.na = p.robot.na
        if start_q is None:
            start_q = 7
        if start_qd is None:
            start_qd = start_q + self.na + 6
        end_q = start_q+self.na
        end_qd = start_qd+self.na
        self.q_slice = slice(start_q, end_q)
        self.qd_slice = slice(start_qd, end_qd)

        self.ddp_counter = -1
        self.interpolate_counter = 0
        self.last = len(self.ddp.us)


    def next(self):
        if self.interpolate_counter % self.window_size == 0:
            self.interpolate_counter = 0
            self.ddp_counter += 1

        if self.ddp_counter >= self.last:
            q_des = self.ddp.xs[self.last][self.q_slice]
            qd_des = np.zeros(self.na)
            tau_ffwd = self.ddp.us[self.last-1]
        elif self.ddp_counter == 0:
            q_des = self.ddp.xs[0][self.q_slice]
            qd_des = self.ddp.xs[0][self.qd_slice]
            tau_ffwd = self.ddp.us[0]

        else:
            q_des_pre = self.ddp.xs[self.ddp_counter-1][self.q_slice]
            qd_des_pre = self.ddp.xs[self.ddp_counter-1][self.qd_slice]
            tau_ffwd_pre = self.ddp.us[self.ddp_counter-1]
            if len(tau_ffwd_pre) == 0:
                tau_ffwd_pre = self.ddp.us[self.ddp_counter - 2]

            q_des_post = self.ddp.xs[self.ddp_counter][self.q_slice]
            qd_des_post = self.ddp.xs[self.ddp_counter][self.qd_slice]
            tau_ffwd_post = self.ddp.us[self.ddp_counter]
            if len(tau_ffwd_post) == 0:
                tau_ffwd_post = self.ddp.us[self.ddp_counter - 1]

            coeff = self.alpha * self.interpolate_counter

            q_des = (1 - coeff) * q_des_pre + coeff * q_des_post
            qd_des = (1 - coeff) * qd_des_pre + coeff * qd_des_post
            tau_ffwd = (1 - coeff) * tau_ffwd_pre + coeff * tau_ffwd_post

        self.interpolate_counter += 1

        return q_des, qd_des, tau_ffwd



