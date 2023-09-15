import numpy as np
import pinocchio as pin

def posePlaneSE3(pose, slope):
    pos = pose[0:3]
    rpy = pose[3:6]

    #pos += np.matmul(pin.rpy.rpyToMatrix(rpy), np.array([1, 0, 0])) # the base of the plane is 2x2

    rpy[1] += slope
    R = pin.rpy.rpyToMatrix(rpy)

    return pos, R


params = {}

# VISUALIZATION PARAMETER
# set useGazebo to:
# True for test the controller in Gazebo environment
# False for visualize the result in Gepetto-GUI (open a terminal and run Gepetto-gui before execute the python script)
params['useGazebo'] = True
#params['useGazebo'] = False

#GROUND PARAMETERS
# check /home/froscia/ros_ws/src/locosim/ros_impedance_controller/worlds/xxx.world to get these poses
# (expressed wrt world frame)
p_Pl0 = np.zeros(3)
R_Pl0 = np.eye(3)
Pl0 = pin.SE3(R_Pl0, p_Pl0)
params['Pl0'] = Pl0

# I prefer to descriver the pose of the plane with a frame having:
# - the x-axis lying on the sloped surface and pointing in the direction of increasing height
# - the y-axis lying on the intersection between the ground floor and the sloped plane
# - the z-axis normal to the plane, and pointing upward
# In this way:
# the origin lies on the x-axis of the world frame,
# the y-axis is parallel to the y-axis of the world frame if yaw angle = 0
# the x(z)-axis are tilted of a slope angle wrt the x(z)-axis of the world frame

# To describe this pose, take the pose=[x, y, z, R, P, Y] parameter in xxx.world (pose of the "midpoint of the plane
# base" frame expressed wrt world frame) and call planeFrameSE3(pose, slope)
params['slop_deg'] = "05"
slope_angle_Pl1 = int(params['slop_deg']) * (np.pi/180)
p_Pl1, R_Pl1 = posePlaneSE3(np.array([1.3, 0, 0, 0, 0, 0]), slope_angle_Pl1)
Pl1 = pin.SE3(R_Pl1, p_Pl1)
params['Pl1'] = Pl1


params['Pl0'] = Pl1
# ROBOT PARAMETERS
# If useFlywheels is set to True, the configuration variables are 19 (pose [7] + leg joints [12] + flywheel position [4])
params['useFlywheels'] = False


# Here (and ONLY here) the initial is expressed wrt the PLANE frame, which has:
# - the origin on the origin of the world frame
# - the z axis on the normal vector to the contact surface
# - the x axis lies on the contact surface and points to the forward direction of the robot
# - the y axis forms a basis for the contact plane and makes the PLANE frame right
# You must transform q0 in the task definition below
if not params['useFlywheels']:
    params['q0'] = np.array([ 0. ,       0.,    0.223,
                              0. ,       0.,    0.   ,     1.,
                              0.2,  np.pi/4, -np.pi/2,
                              0.2, -np.pi/4,  np.pi/2,
                             -0.2,  np.pi/4, -np.pi/2,
                             -0.2, -np.pi/4,  np.pi/2])
else:
    params['q0'] = np.array([ 0. ,       0.,    0.223,
                              0. ,       0.,       0.,     1.,
                              0.2,  np.pi/4, -np.pi/2,
                              0.2, -np.pi/4,  np.pi/2,
                            - 0.2,  np.pi/4, -np.pi/2,
                            - 0.2, -np.pi/4,  np.pi/2,
                              0. ,       0.,       0.,      0.])

# TASK PARAMETERS
#Single Jump on Spot
# approx way of computing flyingKnots:
# Delta_z

# params['GAITPHASES'] = [{
#    'jumping': {
#        'jumpHeight': .3,
#        'jumpLength': [0., 0., 0.],
#        'timeStep': 2e-3,
#        'groundKnots': 150,
#        'flyingKnots': 100,
#        'landingController': False# True
#    }
# }]

# params['GAITPHASES'] = [{
#    'jumping_limited': {
#        'jumpHeight': .1,
#        'jumpLength': [0., 0., 0.],
#        'timeStep': 1e-3,
#        'groundKnots': 1000,
#        'flyingKnots': 500,
#        'landingController': False# True
#    }
# }]

#Multiple Jumps on Spot
# params['GAITPHASES'] = [{
#    'jumping': {
#        'jumpHeight': .3,
#        'jumpLength': [0., 0., 0.],
#        'timeStep': 1e-3,
#        'groundKnots': 100,
#        'flyingKnots': 200
#    }
# }]*3

# Single Forward Jump
# params['GAITPHASES'] = [{
#    'jumping': {
#        'jumpHeight': .6,
#        'jumpLength': [0., 0., 0.],
#        'timeStep': 1e-3,
#        'groundKnots': 100,
#        'flyingKnots': 200
#    }
# }]

params['GAITPHASES'] = [{
    'walking': {
        'stepLength': 0.1,
        'stepHeight': 0.1,
        'timeStep': 2e-3,
        'stepKnots': 100//2,
        'supportKnots': 20//2
    }
}]

# Multiple Forward/Lateral Jump
# params['GAITPHASES'] = [{
#    'jumping': {
#        'jumpHeight': .5,
#        'jumpLength': [0.1, 0.1, 0.],
#        'timeStep': 1e-3,
#        'groundKnots': 100,
#        'flyingKnots': 200
#    }
# }]*3
#
#
# Wobbling
# params['GAITPHASES'] = [{
#          'wobbling': {
#          'timeStep': 1e-2,
#          'frequency': 10,
#          'amplitudes': [np.pi/3, 0, 0],
#          'totalKnots': 500,
#     }
# }]

# Jumping+yaw
# params['GAITPHASES'] = [{
#      'jumping+yaw': {
#          'jumpHeight': .1,
#          'jumpLength': [0., 0., 0],
#          'yaw': -np.pi/6,
#          'timeStep': 1e-3,
#          'groundKnots': 250,
#          'flyingKnots': 300
#      }
# }]

# Multiple Jumping+yaw
# params['GAITPHASES'] = [{
#      'jumping+yaw': {
#          'jumpHeight': .1,
#          'jumpLength': [0., 0., 0],
#          'yaw': -np.pi/6,
#          'timeStep': 1e-3,
#          'groundKnots': 250,
#          'flyingKnots': 300
#      }
# },{
#      'jumping+yaw': {
#          'jumpHeight': .1,
#          'jumpLength': [0., 0., 0],
#          'yaw': np.pi/6,
#          'timeStep': 1e-3,
#          'groundKnots': 250,
#          'flyingKnots': 300
#      }
# }, ]

# Multiple Jumping+yaw
# params['GAITPHASES'] = [{
#      'jumping+yaw': {
#          'jumpHeight': .1,
#          'jumpLength': [0., 0., 0],
#          'yaw': -np.pi/6,
#          'timeStep': 1e-2,
#          'groundKnots': 10,
#          'flyingKnots': 20
#      }
# }]

#Jumping+slope
# quat0 = pin.Quaternion(params['Pl0'].rotation)
#
# params['q0'][0:3] = np.matmul(params['Pl0'].rotation, params['q0'][0:3]) # or R0.T ?
#
# params['q0'][3] = quat0.x
# params['q0'][4] = quat0.y
# params['q0'][5] = quat0.z
# params['q0'][6] = quat0.w
#
#
# #TODO : TO BE FIXED
# p_lf_init =  0.1946
# p_lh_init = -0.1946
# jLx = 0.1
#
# params['GAITPHASES'] = []
# for k in range(0, 1):
#     p_lf0 = p_lf_init + k * jLx
#     p_lh0 = p_lh_init + k * jLx
#     if p_lf0 < 0.5:
#         Pf0 = Ph0 = 0
#     elif p_lf0 >= 0.5 and p_lh0 < 0.5:
#         Pf0 = -5*np.pi/180
#         Ph0 = 0
#     elif p_lh0 >= 0.5:
#         Pf0 = Ph0 = -5*np.pi/180
#
#     p_lf1 = p_lf_init + (k+1) * jLx
#     p_lh1 = p_lh_init + (k+1) * jLx
#     if p_lf1 < 0.5:
#         Pf1 = Ph1 = 0
#     elif p_lf1 >= 0.5 and p_lh1 < 0.5:
#         Pf0 = -5 * np.pi / 180
#         Ph1 = 0
#     elif p_lh1 >= 0.5:
#         Pf1 = Ph1 = -5*np.pi/180
#
#     print(jLx*np.sin(-Pf0))
#     params['GAITPHASES'].append({'jumping+slope': {
#                                  'jumpHeight': .15,
#                                  'jumpLength': [jLx, 0., jLx*np.sin(-Pf0)],
#                                  'timeStep': 1e-3,
#                                  'groundKnots': 100,
#                                  'flyingKnots': 200,
#                                  'RotPlane0': [pin.rpy.rpyToMatrix(0, Pf0, 0),
#                                                pin.rpy.rpyToMatrix(0, Pf0, 0),
#                                                pin.rpy.rpyToMatrix(0, Ph0, 0),
#                                                pin.rpy.rpyToMatrix(0, Ph0, 0)],
#                                  'RotPlane1': [pin.rpy.rpyToMatrix(0, Pf1, 0),
#                                                pin.rpy.rpyToMatrix(0, Pf1, 0),
#                                                pin.rpy.rpyToMatrix(0, Ph1, 0),
#                                                pin.rpy.rpyToMatrix(0, Ph1, 0)]
#                                 }
#                             })
#
#
# params['GAITPHASES'] = [{
#      'jumping+slope': {
#          'jumpHeight': .1,
#          'jumpLength': [0.0, 0., 0.],
#          'timeStep': 1e-3,
#          'groundKnots': 400,
#          'flyingKnots': 150,
#          'RotPlane0': [pin.rpy.rpyToMatrix(0, 0, 0)]*4,
#          'RotPlane1': [pin.rpy.rpyToMatrix(0, 0, 0)]*4
#      }
# },]
# {
#      'jumping+slope': {
#          'jumpHeight': .3,
#          'jumpLength': [0.4, 0., np.tan(5*(np.pi/180))],
#          'timeStep': 1e-3,
#          'groundKnots': 100,
#          'flyingKnots': 200,
#          'RotPlane0': pin.rpy.rpyToMatrix(0, -5*(np.pi/180), 0),
#          'RotPlane1': pin.rpy.rpyToMatrix(0, -5*(np.pi/180),0)
#      }
# }



# # CONTROL PARAMETERS
# # PID
params['kp'] = 5
params['kd'] = .1
params['ki'] = None

# #Landing Controller
