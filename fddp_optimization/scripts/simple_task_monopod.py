'''
    This file uses ddp for generating trajectories to achieve the jump described in xxx.yaml
    The ZMP-based landing controller is NOT used
    The flag USE_REAL_ROBOT must be set to true for experiments, false for simulations
'''

import matplotlib
import pinocchio
import rospkg
from base_controllers.utils.common_functions import plotJoint, plotFrameLinear
import scipy.io.matlab as mio

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(linewidth=np.inf,           # number of characters per line befor new line
                    floatmode='fixed',          # print fixed numer of digits ...
                    precision=8,                # ... 8
                    sign=' ',                   # print space if sign is plus
                    suppress=True,              # suppress scientific notation
                    threshold=np.inf)

import os
import yaml
import pinocchio as pin
from base_controllers.jumpleg_controller import JumpLegController
import  base_controllers.params as conf
from pysolo.controllers.ddp_utils import DDPInterpolate
from pysolo.controllers.quadruped.ddp_monopod import DDPMonopodRobot
from base_controllers.utils.custom_robot_wrapper import RobotWrapper
import rospy as ros
import pandas as pd
from termcolor import colored

USE_REAL_ROBOT = False
ROBOT_NAME = 'jumpleg'                         # go1, solo, (aliengo)
CONFIG_NAME = 'jumping'+ '_' + ROBOT_NAME  # jumping, wobbling, pushup


def computeOptimization(p, problemDescription):
    ################
    # DDP PROBLEM #
    ###############
    # modopod robot
    ddp_monopodRob = DDPMonopodRobot(p.robotPinocchio, 'lf_foot', 'base_link')
    ddp = ddp_monopodRob.generateProblem(x0, problemDescription)

    # Added the callback functions
    print('*** SOLVE Problem: ' + problemDescription['type'] + '***')
    # Solving the problem with the DDP solver
    xs = [x0] * (ddp.problem.T + 1)
    us = ddp.problem.quasiStatic([x0] * ddp.problem.T)
    converged = ddp.solve(xs, us, 250, False, 0.1)
    print('Converged?', converged)
    # TODO show intermediate results with callback

    return ddp

def plotOptimization(p, ddp):
    knot_number_us = len(ddp.us)
    knot_number_xs = len(ddp.xs)
    p.w_x_ee_log = np.empty((3, knot_number_xs))* np.nan
    p.w_base_log = np.empty((3, knot_number_xs)) * np.nan
    p.w_grf_log = np.empty((3, knot_number_xs)) * np.nan
    time_log = np.empty((knot_number_xs)) * np.nan
    time = 0
    for i in range(knot_number_xs-10):
        q_des = ddp.xs[i][:6]
        p.w_base_log[:,i] = ddp.xs[i][:3]
        p.w_x_ee_log[:, i] = p.robot.framePlacement(q_des, p.robot.model.getFrameId(conf.robot_params[p.robot_name]['ee_frame'])).translation
        # J = p.robot.frameJacobian(q_des, p.robot.model.getFrameId(conf.robot_params[p.robot_name]['ee_frame']), True, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, 3:]

        pin.forwardKinematics(p.robot.model, p.robot.data, q_des, np.zeros(p.robot.model.nv),
                              np.zeros(p.robot.model.nv))
        pin.computeJointJacobians(p.robot.model, p.robot.data)
        pin.computeFrameJacobian(p.robot.model, p.robot.data, q_des,
                                 p.robot.model.getFrameId(conf.robot_params[p.robot_name]['ee_frame']))
        J = pin.getFrameJacobian(p.robot.model, p.robot.data,
                                 p.robot.model.getFrameId(conf.robot_params[p.robot_name]['ee_frame']),
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, 3:]


        if len(ddp.us[i]) != 0:
            p.w_grf_log[:, i] = -np.linalg.inv(J.T).dot(ddp.us[i])
        time_log[i] = time
        time+=p.loop_time

def computeFootTargetError(p):
    # compoute error
    # consider size of the foot
    #  rel_error = np.linalg.norm(des_foot_location - (p.w_x_ee+ np.array([0,0,-0.015])) / jump_length
    rel_error = np.linalg.norm(des_foot_location - p.w_x_ee) / jump_length

    # print("Des Foot: ", des_foot_location)
    print("Landing location: ", p.w_x_ee)
    print("Relative Error: ", rel_error)
    return rel_error, p.w_x_ee

def computeBaseTargetError(p):
    # compoute error
    base_pos_w = p.base_offset + p.q[:3]
    rel_error = np.linalg.norm(p.target_CoM - base_pos_w) / jump_length
    print("Base Relative Error: ", rel_error)
    p.rel_error_log.append(rel_error)

if __name__ == '__main__':
    with open(os.environ["PYSOLO_FROSCIA"] + '/pysolo/jumps_configs/' + CONFIG_NAME + '.yaml') as stream:
    #config_file = open(os.environ["PYSOLO_FROSCIA"] + '/pysolo/jumps_configs/' + CONFIG_NAME + '.yaml')
        problemDescription = yaml.load(stream, Loader=yaml.FullLoader)

    DEBUG = False
    test_points = np.loadtxt(os.environ["PYSOLO_FROSCIA"] + '/scripts/test_points.txt')

    p = JumpLegController(ROBOT_NAME)

    xacro_path = rospkg.RosPack().get_path('jumpleg_description') + '/urdf/jumpleg_crocoddyl.xacro'
    args = xacro_path + ' --inorder -o ' + os.environ[
        'LOCOSIM_DIR'] + '/robot_urdf/generated_urdf/jumpleg_crocoddyl.urdf'
    os.system("rosrun xacro xacro " + args)
    urdf_location = os.environ['LOCOSIM_DIR'] + '/robot_urdf/generated_urdf/jumpleg_crocoddyl.urdf'

    p.robotPinocchio = RobotWrapper.BuildFromURDF(urdf_location, root_joint=pinocchio.JointModelTranslation())

    p.start()
    p.startSimulator(world_name="jump_platform.world", additional_args = ['gui:=true'])
    p.loadModelAndPublishers()
    p.initVars()
    p.startupProcedure()
    p.loop_time = conf.robot_params[p.robot_name]['dt']
    landing_position = None
    # loop frequency
    rate = ros.Rate(1 / conf.robot_params[p.robot_name]['dt'])

    #################
    # INITIAL STATE #
    #################
    q0 = np.copy(p.q_des_q0)
    v0 = pin.utils.zero(p.robotPinocchio.model.nv)
    x0 = np.concatenate([q0, v0])


    try:
        df = pd.read_csv('test_optim.csv',header=0)
    except:
        print(colored('CREATING NEW CSV', 'blue'))
        df = pd.DataFrame(columns=['test_nr', 'n_iter', 'target','error_log', 'landing_position','elapsed_time'])

    # DEBUG
    if DEBUG: # overwrite
        df = pd.DataFrame(columns=['test_nr', 'n_iter', 'target','error_log', 'landing_position','elapsed_time'])
        test_points = np.array([[0.3, 0.0, 0.25]])


    for test in range(len(df), test_points.shape[0]):

        # Reset Simulation freezebase does not work
        p.resetBase()
        p.freezeBaseFlag = True
        p.detectedApexFlag = False
        p.detectedTouchDown = False
        p.time = 0
        counter = 0
        p.updateKinematicsDynamics()
        p.tau_ffwd = np.zeros(6)  # base is underactuated

        # #overwrites problem description comment these lines for one test
        problemDescription['callbacks']['verbose'] = False
        # this is the foot position
        problemDescription['jumpLength'] = np.array([test_points[test, 0], test_points[test, 1], test_points[test, 2] -0.25])

        print("new target", problemDescription['jumpLength'])

        des_foot_location = np.array(problemDescription['jumpLength'])
        jump_length = np.linalg.norm(des_foot_location)
        p.target_CoM = des_foot_location.copy()
        p.target_CoM[2] += 0.25
        try:
            print("*** compute optim  *** #:", test)
            p.pause_physics_client()
            import time
            startTime = time.time()
            ddp = computeOptimization(p, problemDescription)
            endTime = time.time()
            elapsed_time = endTime-startTime
            if DEBUG:
                print("elapsed time : ", elapsed_time)
            ddp_ref_gen = DDPInterpolate(p, ddp, problemDescription['timeStep'], p.loop_time, start_q=0, start_qd=6)
            plotOptimization(p, ddp)

            p.unpause_physics_client()
            p.freezeBase(False)

            #print("*** Starting task  ***", p.time)
            p.qd_des_old = np.zeros((6))

            while not ros.is_shutdown():
                p.updateKinematicsDynamics()

                # set references interpolated
                if counter > len(ddp.us):  # us has N element xs N+1
                    # computeFootTargetError(p)
                    # computeBaseTargetError(p)
                    break
                #p.q_des, p.qd_des, p.tau_ffwd[3:] = ddp_ref_gen.next()

                # # otherwise without interp if the controller rate is the same as the ddp rate
                if counter < len(ddp.us):# us has N element xs N+1
                    p.q_des = ddp.xs[counter][:6]
                    p.qd_des = ddp.xs[counter][6:12]
                    if counter>0:
                        p.qdd_des = 1/p.loop_time*(p.qd_des - p.qd_des_old)
                        p.qd_des_old = p.qd_des
                    if len(ddp.us[counter]) !=0:
                        p.tau_ffwd[3:] = ddp.us[counter]
                # to debug
                # p.q_des = p.q_des_q0.copy()
                #p.tau_ffwd[3:] = - p.J.T.dot(p.g[:3]+ p.robot.robot_mass*p.qdd_des[:3])


                if p.time > problemDescription['launchingT']:
                    p.detectApex()
                    if (p.detectedApexFlag):
                        # set jump position (avoid collision in jumping)
                        if not p.detectedTouchDown and p.detectTouchDown():
                            # we compare with the foot error, at the end it will converge to that at the end of the transient
                            rel_error, landing_position = computeFootTargetError(p)
                            #computeBaseTargetError(p)
                            p.detectedTouchDown = True



                # finally, send commands
                p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
                p.logData()

                # plot end-effector and contact force
                if not p.use_ground_truth_contacts:
                    p.ros_pub.add_arrow(
                        p.w_x_ee, p.contactForceW / (10 * p.robot.robot_mass), "green")
                else:
                    p.ros_pub.add_arrow(
                        p.w_x_ee, p.contactForceW / (10 * p.robot.robot_mass), "red")
                p.ros_pub.add_arrow( p.w_x_ee,p.w_grf_log[:,counter], "blue")
                # plot end-effector
                p.ros_pub.add_marker(des_foot_location, color="blue", radius=0.1)
                if landing_position is not None:
                    p.ros_pub.add_marker(landing_position, color="red", radius=0.15)
                p.ros_pub.add_marker( p.w_x_ee, radius=0.05)
                p.ros_pub.add_cone(p.w_x_ee, np.array([0, 0, 1.]),
                                   problemDescription['mu'], height=0.15, color="blue")

                p.ros_pub.publishVisual()
                # wait for synconization of the control loop
                rate.sleep()

                counter += 1
                p.time = np.round(p.time + np.array([p.loop_time]), 3)  # to avoid issues of dt 0.0009999

            data = {'test_nr':test,'n_iter':ddp.iter,'target':p.target_CoM, 'error_log':rel_error, 'landing_position': landing_position, 'elapsed_time':elapsed_time}
            df = df.append(data, ignore_index=True)
            df.to_csv('test_optim.csv', index=None)


        except (ros.ROSInterruptException, ros.service.ServiceException):
            ros.signal_shutdown("killed")
            p.deregister_node()

        # DEBUG
        if DEBUG:
            plotJoint('position', time_log = p.time_log.flatten(), q_log = p.q_log, q_des_log = p.q_des_log, joint_names=p.joint_names, sharey=False)
            #plotJoint('velocity', time_log = p.time_log.flatten(), qd_log=p.qd_log, qd_des_log=p.qd_des_log,joint_names=p.joint_names, sharey=False)
            plotJoint('torque', time_log = p.time_log.flatten(), tau_ffwd_log = p.tau_ffwd_log, tau_log = p.tau_log, joint_names=p.joint_names, sharey=False)
            plotFrameLinear('position', time_log=p.time_log.flatten(), Pose_log=p.contactForceW_log,
                      title='grf', frame='W', sharex=True, sharey=False, start=0, end=-1)
            plotFrameLinear('position', time_log=p.time_log.flatten(), Pose_log=p.w_x_ee_log,
                        title='foot pos', frame='W', sharex=True, sharey=False, start=0, end=-1)

    ros.signal_shutdown("killed")
    p.deregister_node()