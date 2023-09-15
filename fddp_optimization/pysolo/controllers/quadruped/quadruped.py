import crocoddyl
import numpy as np
import scipy.linalg as l
import pinocchio as pin
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem


class QuadrupedalGaitProblem(SimpleQuadrupedalGaitProblem):
    def __init__(self, robot, lfFoot, rfFoot, lhFoot, rhFoot, base, q0):
        # Robot
        self.robot = robot
        self.rmodel = self.model = self.robot.model
        self.rdata = self.data = self.robot.data  # TODO: check this line (possibly, replace it with self.rdata = self.rmodel.ceateData())
        # Dynamics
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # actuation bounds
        self.actuation.lb = -self.rmodel.effortLimit
        self.actuation.ub = self.rmodel.effortLimit

        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        # Defining default state
        self.rmodel.q0 = q0
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 1.0
        self.Rsurf = self.Rsurf0 = np.eye(3)
        self.Rsurf_final = np.eye(3)
        # List of rotation matrices at the contact points (should be changed at run time)
        self.Rsurf_list = [np.eye(3)]*4

        # RotPlane0 = self.Rsurf0
        # RotPlane1 = self.Rsurf_final

        self.lfFootPosHist = []
        self.comPosHist = []
        self.baseRotHist = []

        self.Rbase_0 = pin.Quaternion(self.rmodel.q0[3:7]).toRotationMatrix()

        self.baseId = self.rmodel.getFrameId(base)

        self.euler = []
        self.angle = []
        self.axis = []

    def createBaseWobblingProblem(self, x0, timeStep, frequency, amplitudes, totalKnots):
        self.rmodel.defaultState = x0
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        baseRot0 = self.rdata.oMf[self.baseId].rotation
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.

        loco3dModel = []
        wobblingPhase = []

        for k in range(totalKnots):
            delta_rpy = np.array(amplitudes) * np.sin(2*np.pi*frequency * (k + 1) / totalKnots)
            R_t = pin.rpy.rpyToMatrix(delta_rpy) + baseRot0
            model = self.createBaseWobblingModel(timeStep=timeStep,
                                                supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                baseRotTask=crocoddyl.FrameRotation(self.baseId, R_t)
                                                )

            wobblingPhase.append(model)
        loco3dModel += wobblingPhase

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem


    def createBaseWobblingModel(self, timeStep, supportFootIds, baseRotTask=None):
        """ Action model for a swing foot phase.

               :param timeStep: step duration of the action model
               :param supportFootIds: Ids of the constrained feet
               :param comTask: CoM task
               :param swingFootTask: swinging foot task
               :param baseRotationTask: base rotation task
               :return action model for a swing foot phase
               """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        # if swingFootTask is not None:
        #     for i in swingFootTask:
        #         xref = crocoddyl.FrameTranslation(i.id, i.placement.translation)
        #         footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
        #         costModel.addCost(self.rmodel.frames[i.id].name + "_footTrack", footTrack, 1e6)

        if baseRotTask is not None:
            Rref = baseRotTask
            baseTrack = crocoddyl.CostModelFrameRotation(self.state, Rref, self.actuation.nu)
            costModel.addCost(self.rmodel.frames[self.baseId].name + "_baseTrack", baseTrack, 1e6)

            rpy = pin.rpy.matrixToRpy(Rref.rotation)
            aa = pin.AngleAxis(Rref.rotation)

            self.euler.append(rpy)
            self.angle.append(aa.angle)
            self.axis.append(aa.axis)


        stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
                                (self.rmodel.nv - 6))
        stateReg = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBounds = crocoddyl.CostModelState(
            self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
            0 * self.rmodel.defaultState, self.actuation.nu)
        costModel.addCost("stateBounds", stateBounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model




    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pin.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        max_displacement_up = np.array([jumpLength[0]/2, jumpLength[1]/2, jumpLength[2] + jumpHeight])
        max_displacement_down = np.array([jumpLength[0], jumpLength[1], jumpLength[2]])


        loco3dModel = []
        takeOff = [
            self.createFlyingModel(timeStep=timeStep,
                                   supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]

        flyingUpPhase = [
            self.createFlyingModel(timeStep=timeStep,
                                   supportFootIds=[],
                                   comTask = max_displacement_up * ( (k + 1) / flyingKnots) + comRef)
            for k in range(flyingKnots)
        ]

        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createFlyingModel(timeStep=timeStep,
                                                       supportFootIds=[],
                                                       comTask=max_displacement_down * (k + 1) / flyingKnots + max_displacement_up * (1 - (k + 1) / flyingKnots) + comRef)]

        f0 = jumpLength.copy()
        footTask = [[self.lfFootId, pin.SE3(np.eye(3), lfFootPos0 + f0)],
                    [self.rfFootId, pin.SE3(np.eye(3), rfFootPos0 + f0)],
                    [self.lhFootId, pin.SE3(np.eye(3), lhFootPos0 + f0)],
                    [self.rhFootId, pin.SE3(np.eye(3), rhFootPos0 + f0)]]
        landingPhase = [
            self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
        ]
        f0[2] = df
        landed = [
            self.createFlyingModel(timeStep=timeStep,
                                   supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                   comTask=comRef + f0)
            for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createJumpingYawProblem(self, x0, jumpHeight, jumpLength, yaw, timeStep, groundKnots, flyingKnots):
        self.rmodel.defaultState = x0
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        baseRot0 = self.rdata.oMi[1].rotation

        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = np.asscalar(pin.centerOfMass(self.rmodel, self.rdata, q0)[2])

        loco3dModel = []


        ### TAKE OFF ###
        takeOff = [ ]
        for k in range(groundKnots):
            model = self.createFlyingModel( timeStep=timeStep,
                                            supportFootIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                            baseRotTask = crocoddyl.FrameRotation(self.baseId, baseRot0)
                                          )
            takeOff.append(model)


        ### FLYING UP PHASE ###
        com_up = np.array([jumpLength[0]/2, jumpLength[1]/2, jumpLength[2] + jumpHeight])
        axis = np.array([0,0,1])

        flyingUpPhase = []
        for k in range(flyingKnots):
            Rt = slerp_aa(baseRot0, yaw/2, axis, (k + 1) / flyingKnots)
            delta_com = com_up *  (k + 1) / flyingKnots

            Rf = np.matmul(Rt, baseRot0.T)

            lfFootPos = np.matmul(Rf, lfFootPos0 + delta_com)
            rfFootPos = np.matmul(Rf, rfFootPos0 + delta_com)
            lhFootPos = np.matmul(Rf, lhFootPos0 + delta_com)
            rhFootPos = np.matmul(Rf, rhFootPos0 + delta_com)


            footTask = [
                crocoddyl.FramePlacement(self.lfFootId, pin.SE3(Rf, lfFootPos)),
                crocoddyl.FramePlacement(self.rfFootId, pin.SE3(Rf, rfFootPos)),
                crocoddyl.FramePlacement(self.lhFootId, pin.SE3(Rf, lhFootPos)),
                crocoddyl.FramePlacement(self.rhFootId, pin.SE3(Rf, rhFootPos))
            ]

            print('lf foot = ', pin.SE3(Rf, lfFootPos))

            baseRotTask = crocoddyl.FrameRotation(self.baseId, Rt)
            flyingUpPhase.append(self.createFlyingModel(timeStep=timeStep,
                                   supportFootIds = [],
                                   swingFootTask = footTask,
                                   comTask = np.matmul(Rt, delta_com + comRef),
                                   baseRotTask = baseRotTask
                                   )
                                 )

        ### FLYING DOWN PHASE ###
        baseRot_up = slerp_aa(baseRot0, yaw / 2, axis, 1)
        com_down = np.array([jumpLength[0], jumpLength[1], jumpLength[2]])
        flyingDownPhase = []

        for k in range(flyingKnots):
            Rt = slerp_aa(baseRot_up, yaw/2, axis, (k + 1) / flyingKnots)
            delta_com = com_up - (com_up - com_down) * (k + 1) / flyingKnots

            Rf = np.matmul(Rt, baseRot0.T)

            lfFootPos = np.matmul(Rf, lfFootPos0 + delta_com)
            rfFootPos = np.matmul(Rf, rfFootPos0 + delta_com)
            lhFootPos = np.matmul(Rf, lhFootPos0 + delta_com)
            rhFootPos = np.matmul(Rf, rhFootPos0 + delta_com)

            footTask = [
                crocoddyl.FramePlacement(self.lfFootId, pin.SE3(Rf, lfFootPos)),
                crocoddyl.FramePlacement(self.rfFootId, pin.SE3(Rf, rfFootPos)),
                crocoddyl.FramePlacement(self.lhFootId, pin.SE3(Rf, lhFootPos)),
                crocoddyl.FramePlacement(self.rhFootId, pin.SE3(Rf, rhFootPos))
            ]

            print('lf foot = ', pin.SE3(Rf, lfFootPos))

            baseRotTask = crocoddyl.FrameRotation(self.baseId, Rt)
            flyingDownPhase.append(self.createFlyingModel(timeStep=timeStep,
                                                          supportFootIds = [],
                                                          swingFootTask = footTask,
                                                          comTask = np.matmul(Rt, delta_com + comRef),
                                                          baseRotTask = baseRotTask
                                                         )
                                 )

        ### LANDING PHASE ###
        baseRot_down = slerp_aa(baseRot_up, yaw/2, axis, 1)
        Rf = np.matmul(baseRot_down, baseRot0.T)
        f0 = jumpLength
        lfFootPosf = np.matmul(Rf, lfFootPos0+f0)
        rfFootPosf = np.matmul(Rf, rfFootPos0+f0)
        lhFootPosf = np.matmul(Rf, lhFootPos0+f0)
        rhFootPosf = np.matmul(Rf, rhFootPos0+f0)

        footTask = [[self.lfFootId, pin.SE3(Rf, lfFootPosf)],
                    [self.rfFootId, pin.SE3(Rf, rfFootPosf)],
                    [self.lhFootId, pin.SE3(Rf, lhFootPosf)],
                    [self.rhFootId, pin.SE3(Rf, rhFootPosf)]
        ]
        landingPhase = [
            self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
        ]

        ### LANDED PHASE ###
        f0[2] = df
        landed = [
            self.createFlyingModel(timeStep = timeStep,
                                   supportFootIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                   comTask=np.matmul(baseRot_down, comRef + f0),
                                   baseRotTask=crocoddyl.FrameRotation(self.baseId,baseRot_down))
            for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createJumpingOnSlopedPlaneOLD(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots, RotPlane0, RotPlane1):
        self.rmodel.defaultState = x0
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        baseRot0 = self.rdata.oMi[1].rotation

        aa = pin.AngleAxis(np.matmul(RotPlane0.T, RotPlane1))
        angle = aa.angle
        axis = aa.axis

        df = jumpLength[2] - rfFootPos0[2]

        # We assume the quadruped hardly symmetric
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = np.asscalar(pin.centerOfMass(self.rmodel, self.rdata, q0)[2])

        loco3dModel = []

        ### TAKE OFF ###
        takeOff = []
        for k in range(groundKnots):
            model = self.createFlyingModel(timeStep=timeStep,
                                           supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                           baseRotTask=crocoddyl.FrameRotation(self.baseId, baseRot0)
                                           )
            takeOff.append(model)

        ### FLYING UP PHASE ###
        com_up = np.array([jumpLength[0] / 2, jumpLength[1] / 2, jumpLength[2] + jumpHeight])

        upPhaseKnots = int(0.5 * 2) * flyingKnots #TODO: a better approach could consider different time length for flying up & down phases

        flyingUpPhase = []
        for k in range(upPhaseKnots):
            Rt = slerp_aa(RotPlane0, angle / 2, axis, (k + 1) / upPhaseKnots)
            delta_com = com_up * (k + 1) / upPhaseKnots

            Rf = np.matmul(Rt, RotPlane0.T)

            lfFootPos = np.matmul(Rf, lfFootPos0 + delta_com)
            rfFootPos = np.matmul(Rf, rfFootPos0 + delta_com)
            lhFootPos = np.matmul(Rf, lhFootPos0 + delta_com)
            rhFootPos = np.matmul(Rf, rhFootPos0 + delta_com)


            footTask = [
                crocoddyl.FramePlacement(self.lfFootId, pin.SE3(Rf, lfFootPos)),
                crocoddyl.FramePlacement(self.rfFootId, pin.SE3(Rf, rfFootPos)),
                crocoddyl.FramePlacement(self.lhFootId, pin.SE3(Rf, lhFootPos)),
                crocoddyl.FramePlacement(self.rhFootId, pin.SE3(Rf, rhFootPos))
            ]
            self.lhFootPosHist += [lhFootPos]

            baseRotTask = crocoddyl.FrameRotation(self.baseId, Rt)
            flyingUpPhase.append(self.createFlyingModel(timeStep=timeStep,
                                                        supportFootIds=[],
                                                        swingFootTask=footTask,
                                                        comTask=np.matmul(Rt, delta_com + comRef),
                                                        baseRotTask=baseRotTask
                                                        )
                                 )

        ### FLYING DOWN PHASE ###
        baseRot_up = slerp_aa(baseRot0, angle / 2, axis, 1)
        com_down = np.array([jumpLength[0], jumpLength[1], jumpLength[2]])
        flyingDownPhase = []

        downPhaseKnots = int(0.5 * 2) * flyingKnots  # TODO: a better approach could consider different time length for flying up & down phases

        for k in range(downPhaseKnots):
            Rt = slerp_aa(baseRot_up, angle / 2, axis, (k + 1) / downPhaseKnots)
            delta_com = com_up - (com_up - com_down) * (k + 1) / downPhaseKnots

            Rf = np.matmul(Rt, RotPlane0.T)

            lfFootPos = np.matmul(Rf, lfFootPos0 + delta_com)
            rfFootPos = np.matmul(Rf, rfFootPos0 + delta_com)
            lhFootPos = np.matmul(Rf, lhFootPos0 + delta_com)
            rhFootPos = np.matmul(Rf, rhFootPos0 + delta_com)

            footTask = [
                crocoddyl.FramePlacement(self.lfFootId, pin.SE3(Rf, lfFootPos)),
                crocoddyl.FramePlacement(self.rfFootId, pin.SE3(Rf, rfFootPos)),
                crocoddyl.FramePlacement(self.lhFootId, pin.SE3(Rf, lhFootPos)),
                crocoddyl.FramePlacement(self.rhFootId, pin.SE3(Rf, rhFootPos))
            ]

            self.lhFootPosHist += [lhFootPos]

            baseRotTask = crocoddyl.FrameRotation(self.baseId, Rt)
            flyingDownPhase.append(self.createFlyingModel(timeStep=timeStep,
                                                          supportFootIds=[],
                                                          swingFootTask=footTask,
                                                          comTask=np.matmul(Rt, delta_com + comRef),
                                                          baseRotTask=baseRotTask
                                                          )
                                   )

        ### LANDING PHASE ###
        baseRot_down = RotPlane1 # slerp_aa(baseRot_up, yaw / 2, axis, 1)
        Rf = np.matmul(baseRot_down, baseRot0.T)
        f0 = jumpLength
        lfFootPosf = np.matmul(Rf, lfFootPos0 + f0)
        rfFootPosf = np.matmul(Rf, rfFootPos0 + f0)
        lhFootPosf = np.matmul(Rf, lhFootPos0 + f0)
        rhFootPosf = np.matmul(Rf, rhFootPos0 + f0)
        self.lhFootPosHist += [lhFootPosf]

        footTask = [[self.lfFootId, pin.SE3(Rf, lfFootPosf)],
                    [self.rfFootId, pin.SE3(Rf, rfFootPosf)],
                    [self.lhFootId, pin.SE3(Rf, lhFootPosf)],
                    [self.rhFootId, pin.SE3(Rf, rhFootPosf)]
                    ]
        landingPhase = [
            self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
        ]

        ### LANDED PHASE ###
        f0[2] = df
        landed = [
            self.createFlyingModel(timeStep=timeStep,
                                   supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                   comTask=np.matmul(baseRot_down, comRef + f0),
                                   baseRotTask=crocoddyl.FrameRotation(self.baseId, baseRot_down))
            for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem



    def createJumpingOnSlopedPlane(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots, RotPlane0, RotPlane1):
        err_str = 'RotPlane0 and RotPlane1 must be either a 3x3 np.ndarray or a list of 3x3 np.ndarray'
        print(RotPlane0)
        if type(RotPlane0) == np.ndarray and RotPlane0.shape == (3, 3):
            RotPlane0 = [RotPlane0]*4
        elif type(RotPlane0) == list:
            for R in RotPlane0:
                if R.shape != (3, 3):
                    raise Exception(err_str)
            RotPlane0 = RotPlane0
        else:
            raise Exception(err_str)

        if type(RotPlane1) == np.ndarray and RotPlane1.shape == (3, 3):
            RotPlane1 = [RotPlane1]*4
        elif type(RotPlane1) == list:
            for R in RotPlane1:
                if R.shape != (3, 3):
                    raise Exception(err_str)
            RotPlane1 = RotPlane1
        else:
            raise Exception(err_str)

        self.rmodel.defaultState = x0.copy()
        q0 = self.rmodel.defaultState[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        baseRot0 = self.rdata.oMi[1].rotation

        aa = []
        angle = []
        axis = []

        for i in range(len(RotPlane0)):
            aa.append(pin.AngleAxis(np.matmul(RotPlane0[i].T, RotPlane1[i])))
            angle.append(aa[-1].angle)
            axis.append(aa[-1].axis)
        df = jumpLength[2] - rfFootPos0[2]

        # We assume the quadruped hardly symmetric
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = np.asscalar(pin.centerOfMass(self.rmodel, self.rdata, q0)[2])

        loco3dModel = []

        ### TAKE OFF ###
        takeOff = []
        footTask = [
            crocoddyl.FramePlacement(self.lfFootId, pin.SE3(self.rdata.oMf[self.lfFootId].rotation, lfFootPos0)),
            crocoddyl.FramePlacement(self.rfFootId, pin.SE3(self.rdata.oMf[self.rfFootId].rotation, rfFootPos0)),
            crocoddyl.FramePlacement(self.lhFootId, pin.SE3(self.rdata.oMf[self.lhFootId].rotation, lhFootPos0)),
            crocoddyl.FramePlacement(self.rhFootId, pin.SE3(self.rdata.oMf[self.rhFootId].rotation, rhFootPos0))
        ]

        for k in range(groundKnots):
            model = self.createFlyingModel(timeStep=timeStep,
                                           supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                           baseRotTask=crocoddyl.FrameRotation(self.baseId, baseRot0)
                                           )
            takeOff.append(model)
            self.lfFootPosHist.append(lfFootPos0)
            self.comPosHist.append(comRef)
            self.baseRotHist.append(pin.rpy.matrixToRpy(baseRot0))

        ### FLYING UP PHASE ###
        com_up = np.array([jumpLength[0] / 2, jumpLength[1] / 2, jumpLength[2] + jumpHeight])

        upPhaseKnots = int(0.5 * 2) * flyingKnots #TODO: a better approach could consider different time length for flying up & down phases

        flyingUpPhase = []
        for k in range(upPhaseKnots):

            lfRt = slerp_aa(RotPlane0[0], angle[0] / 2, axis[0], (k + 1) / upPhaseKnots)
            rfRt = slerp_aa(RotPlane0[1], angle[1] / 2, axis[1], (k + 1) / upPhaseKnots)
            lhRt = slerp_aa(RotPlane0[2], angle[2] / 2, axis[2], (k + 1) / upPhaseKnots)
            rhRt = slerp_aa(RotPlane0[3], angle[3] / 2, axis[3], (k + 1) / upPhaseKnots)

            baseRott = R_avg_list([rfRt, lfRt, rhRt, lhRt])  # from right to left, from hind to front

            delta_com = com_up * (k + 1) / upPhaseKnots

            lfFootPos = np.matmul(lfRt, lfFootPos0 + delta_com)
            rfFootPos = np.matmul(rfRt, rfFootPos0 + delta_com)
            lhFootPos = np.matmul(lhRt, lhFootPos0 + delta_com)
            rhFootPos = np.matmul(rhRt, rhFootPos0 + delta_com)


            footTask = [
                crocoddyl.FramePlacement(self.lfFootId, pin.SE3(lfRt, lfFootPos)),
                crocoddyl.FramePlacement(self.rfFootId, pin.SE3(rfRt, rfFootPos)),
                crocoddyl.FramePlacement(self.lhFootId, pin.SE3(lhRt, lhFootPos)),
                crocoddyl.FramePlacement(self.rhFootId, pin.SE3(rhRt, rhFootPos))
            ]

            baseRotTask = crocoddyl.FrameRotation(self.baseId, baseRott)
            comTask = np.matmul(baseRott, delta_com + comRef)
            flyingUpPhase.append(self.createFlyingModel(timeStep=timeStep,
                                                        supportFootIds=[],
                                                        swingFootTask=footTask,
                                                        comTask=comTask,
                                                        baseRotTask=baseRotTask
                                                        )
                                 )
            self.lfFootPosHist.append(lfFootPos)
            self.comPosHist.append(comTask)
            self.baseRotHist.append(pin.rpy.matrixToRpy(baseRott))

        ### FLYING DOWN PHASE ###
        baseRot_up = baseRott
        feetRot_up = [lfRt, rfRt, lhRt, rhRt]
        com_down = np.array([jumpLength[0], jumpLength[1], jumpLength[2]])
        flyingDownPhase = []

        downPhaseKnots = int(0.5 * 2) * flyingKnots  # TODO: a better approach could consider different time length for flying up & down phases

        for k in range(downPhaseKnots):
            lfRt = slerp_aa(feetRot_up[0], angle[0] / 2, axis[0], (k + 1) / downPhaseKnots)
            rfRt = slerp_aa(feetRot_up[1], angle[1] / 2, axis[1], (k + 1) / downPhaseKnots)
            lhRt = slerp_aa(feetRot_up[2], angle[2] / 2, axis[2], (k + 1) / downPhaseKnots)
            rhRt = slerp_aa(feetRot_up[3], angle[3] / 2, axis[3], (k + 1) / downPhaseKnots)

            baseRott = R_avg_list([rfRt, lfRt, rhRt, lhRt])  # from right to left, from hind to front

            sigma = (k + 1) / downPhaseKnots
            delta_com = com_up * (1-sigma) + com_down*sigma
            # delta_com = com_up - (com_up - com_down) * (k + 1) / downPhaseKnots

            lfFootPos = np.matmul(lfRt, lfFootPos0 + delta_com)
            rfFootPos = np.matmul(rfRt, rfFootPos0 + delta_com)
            lhFootPos = np.matmul(lhRt, lhFootPos0 + delta_com)
            rhFootPos = np.matmul(rhRt, rhFootPos0 + delta_com)

            footTask = [
                crocoddyl.FramePlacement(self.lfFootId, pin.SE3(lfRt, lfFootPos)),
                crocoddyl.FramePlacement(self.rfFootId, pin.SE3(rfRt, rfFootPos)),
                crocoddyl.FramePlacement(self.lhFootId, pin.SE3(lhRt, lhFootPos)),
                crocoddyl.FramePlacement(self.rhFootId, pin.SE3(rhRt, rhFootPos))
            ]

            comTask = np.matmul(baseRott, delta_com + comRef)
            baseRotTask = crocoddyl.FrameRotation(self.baseId, baseRott)
            flyingDownPhase.append(self.createFlyingModel(timeStep=timeStep,
                                                          supportFootIds=[],
                                                          swingFootTask=footTask,
                                                          comTask=comTask,
                                                          baseRotTask=baseRotTask
                                                          )
                                   )
            self.lfFootPosHist.append(lfFootPos)
            self.comPosHist.append(comTask)
            self.baseRotHist.append(pin.rpy.matrixToRpy(baseRott))

        ### LANDING PHASE ###
        baseRot_down = baseRott # slerp_aa(baseRot_up, yaw / 2, axis, 1)
        #bug here
        # lfRf = np.matmul(lfRt, RotPlane0[0].T)
        # rfRf = np.matmul(rfRt, RotPlane0[1].T)
        # lhRf = np.matmul(lhRt, RotPlane0[2].T)
        # rhRf = np.matmul(rhRt, RotPlane0[3].T)

        lfRf = RotPlane1[0]
        rfRf = RotPlane1[1]
        lhRf = RotPlane1[2]
        rhRf = RotPlane1[3]

        f0 = jumpLength.copy()  # TODO: here it is better to modify f0 for the feet on the ramp
        lfFootPosf = np.matmul(lfRf, lfFootPos0 + f0)
        rfFootPosf = np.matmul(rfRf, rfFootPos0 + f0)
        lhFootPosf = np.matmul(lhRf, lhFootPos0 + f0)
        rhFootPosf = np.matmul(rhRf, rhFootPos0 + f0)

        #baseRott = R_avg_list([rhRf, lhRf, rfRf, lfRf])  # from right to left, from hind to front

        footTask = [[self.lfFootId, pin.SE3(lfRf, lfFootPosf)],
                    [self.rfFootId, pin.SE3(rfRf, rfFootPosf)],
                    [self.lhFootId, pin.SE3(lhRf, lhFootPosf)],
                    [self.rhFootId, pin.SE3(rhRf, rhFootPosf)]
                    ]
        landingPhase = [
            self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
        ]

        self.lfFootPosHist.append(lfFootPosf)
        self.comPosHist.append(comTask)
        self.baseRotHist.append(pin.rpy.matrixToRpy(baseRot_down))

        ### LANDED PHASE ###
        f0[2] = df
        comTask = np.matmul(baseRot_down, com_down+comRef)
        baseRotTask = crocoddyl.FrameRotation(self.baseId, baseRot_down)
        landed = []
        for k in range(groundKnots):
            landed.append(self.createFlyingModel(timeStep=timeStep,
                                                 supportFootIds=[self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                 comTask=np.matmul(baseRot_down, com_down+comRef),
                                                 baseRotTask=baseRotTask
                                                 )
                          )

            self.lfFootPosHist.append(lfFootPosf)
            self.comPosHist.append(comTask)
            self.baseRotHist.append(pin.rpy.matrixToRpy(baseRot_down))

        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(self.rmodel.defaultState, loco3dModel, loco3dModel[-1])
        return problem


















    # TODO
    def createSomersaultProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        baseRot0 = self.rdata.oMf[self.baseId].rotation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = np.asscalar(pin.centerOfMass(self.rmodel, self.rdata, q0)[2])

        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]

        com_up = np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight])

        axis = np.array([0,0,1])

        flyingUpPhase = [
            self.createFlyingModel( timeStep = timeStep,
                                    supportFootIds = [],
                                    comTask = com_up * (k + 1) / flyingKnots + comRef,
                                    baseRotTask = crocoddyl.FrameRotation(self.baseId,
                                                                         slerp_aa(baseRot0, np.pi/2, axis, (k + 1) / flyingKnots))
                                    )
            for k in range(flyingKnots)
        ]

        baseRot_up = slerp_aa(baseRot0, np.pi/2, axis, 1)

        flyingDownPhase = [
            self.createFlyingModel(timeStep=timeStep,
                                   supportFootIds=[],
                                   baseRotTask=crocoddyl.FrameRotation(self.baseId,
                                                                       slerp_aa(baseRot_up, np.pi/2, axis, (k + 1) / flyingKnots))
                                   )
            for k in range(flyingKnots)
        ]

        f0 = jumpLength
        footTask = [[self.lfFootId, pin.SE3(baseRot_down, lfFootPosf + f0)],
                    [self.rfFootId, pin.SE3(baseRot_down, rfFootPosf + f0)],
                    [self.lhFootId, pin.SE3(baseRot_down, lhFootPosf + f0)],
                    [self.rhFootId, pin.SE3(baseRot_down, rhFootPosf + f0)]
                    ]

        landingPhase = [
            self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
        ]
        f0[2] = df
        landed = [
            self.createSwingFootModel(timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                      comTask=comRef + f0) for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem




    def createFlyingModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None, baseRotTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :param baseRotationTask: base rotation task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        flag = False
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)


        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False, 5.)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                xref = crocoddyl.FrameTranslation(i.id, i.placement.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.id].name + "_footTrack", footTrack, 1e3)

        # The cost is defined as 1/2*||r||^2 (default activation model) where
        # r = R⊖R_ref is the residual vector
        # Cost and residual derivatives are computed analytically.
        # For the computation of the cost Hessian, we use the Gauss-Newton approximation
        if baseRotTask is not None:
            Rref = baseRotTask
            baseTrack = crocoddyl.CostModelFrameRotation(self.state, Rref, self.actuation.nu)
            costModel.addCost(self.rmodel.frames[self.baseId].name + "_baseTrack", baseTrack, 1e6)

            rpy = pin.rpy.matrixToRpy(Rref.rotation)
            aa = pin.AngleAxis(Rref.rotation)

            self.euler.append(rpy)
            self.angle.append(aa.angle)
            self.axis.append(aa.axis)



        stateWeights = np.array([0.] * 3 + [0.] * 3 + [0.01] * (self.rmodel.nv - 6) + [0.] * 6 + [1.] *
                                (self.rmodel.nv - 6))

        stateReg = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2),
                                            self.rmodel.defaultState, self.actuation.nu)

        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e2)

        x_lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        x_ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBounds = crocoddyl.CostModelState(
            self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub)),
                                                                  0 * self.rmodel.defaultState, self.actuation.nu)
        costModel.addCost("stateBounds", stateBounds, 1e3)

        ctrlBoundsAcrivationModel = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(self.actuation.lb[-self.actuation.nu:], self.actuation.ub[-self.actuation.nu:]))

        ctrlBounds = crocoddyl.CostModelControl(self.state,
                                                ctrlBoundsAcrivationModel,
                                                self.actuation.nu)

        costModel.addCost("ctrlBounds", ctrlBounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if supportFootIds == []:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                         costModel, 0., False)
        else:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                         costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)

        return model


    def mycreateSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
                                                           np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                if type(i) == type([]):
                    old_i = i
                    i = {}
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i.id,
                                                                                   i.placement.translation, nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
                                (self.rmodel.nv - 6))
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e1)

        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        costModel.addCost("stateBounds", stateBounds, 1e3)

        ctrlBoundsAcrivationModel = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(self.actuation.lb[-self.actuation.nu:], self.actuation.ub[-self.actuation.nu:]))

        ctrlBounds = crocoddyl.CostModelControl(self.state,
                                                ctrlBoundsAcrivationModel,
                                                self.actuation.nu)

        #costModel.addCost("ctrlBounds", ctrlBounds, 1e6)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    # def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
    #     """ Action model for a foot switch phase.
    #
    #     :param supportFootIds: Ids of the constrained feet
    #     :param swingFootTask: swinging foot task
    #     :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
    #     :return action model for a foot switch phase
    #     """
    #     if pseudoImpulse:
    #         return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
    #     else:
    #         return self.createImpulseModel(supportFootIds, swingFootTask)

    # def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
    #     """ Action model for pseudo-impulse models.
    #
    #     A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
    #     :param supportFootIds: Ids of the constrained feet
    #     :param swingFootTask: swinging foot task
    #     :return pseudo-impulse differential action model
    #     """
    #     # Creating a 3D multi-contact model, and then including the supporting
    #     # foot
    #     nu = self.actuation.nu
    #     contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
    #     for i in supportFootIds:
    #         supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
    #                                                        np.array([0., 50.]))
    #         contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)
    #
    #     # Creating the cost model for a contact phase
    #     costModel = crocoddyl.CostModelSum(self.state, nu)
    #     for i in supportFootIds:
    #         cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
    #         coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
    #         coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
    #         frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
    #         costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
    #     if swingFootTask is not None:
    #         for i in swingFootTask:
    #             frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i.id,
    #                                                                                i.placement.translation, nu)
    #             frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state, i[0], pinocchio.Motion.Zero(),
    #                                                                          pinocchio.LOCAL, nu)
    #             footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
    #             impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
    #             costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e7)
    #             costModel.addCost(self.rmodel.frames[i[0]].name + "_impulseVel", impulseFootVelCost, 1e6)
    #
    #     stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * self.rmodel.nv)
    #     stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
    #     stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2)
    #     ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
    #     stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
    #     ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
    #     costModel.addCost("stateReg", stateReg, 1e1)
    #     costModel.addCost("ctrlReg", ctrlReg, 1e-3)
    #
    #     # Creating the action model for the KKT dynamics with simpletic Euler
    #     # integration scheme
    #     dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
    #                                                                  costModel, 0., True)
    #     model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
    #     return model
    #
    # def createImpulseModel(self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0):
    #     """ Action model for impulse models.
    #
    #     An impulse model consists of describing the impulse dynamics against a set of contacts.
    #     :param supportFootIds: Ids of the constrained feet
    #     :param swingFootTask: swinging foot task
    #     :return impulse action model
    #     """
    #     # Creating a 3D multi-contact model, and then including the supporting foot
    #     nu = self.actuation.nu
    #     impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
    #     for i in supportFootIds:
    #         supportContactModel = crocoddyl.ImpulseModel3D(self.state, i)
    #         impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)
    #
    #     # Creating the cost model for a contact phase
    #     costModel = crocoddyl.CostModelSum(self.state, nu)
    #     if swingFootTask is not None:
    #         for i in swingFootTask:
    #             frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i.id,
    #                                                                                i.placement.translation, nu)
    #             footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
    #             costModel.addCost(self.rmodel.frames[i.id].name + "_footTrack", footTrack, 1e7)
    #
    #     stateWeights = np.array([1.] * 6 + [10.] * (self.rmodel.nv - 6) + [10.] * self.rmodel.nv)
    #     stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
    #     stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2)
    #     stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
    #     costModel.addCost("stateReg", stateReg, 1e1)
    #
    #     # Creating the action model for the KKT dynamics with simpletic Euler
    #     # integration scheme
    #     model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
    #     model.JMinvJt_damping = JMinvJt_damping
    #     model.r_coeff = r_coeff
    #     return model




# slerp with angle and axis describing the final orientation
def slerp_aa(R0, angle, axis, t):
    ax_skew = pin.skew(axis)
    ax_skew2 = np.matmul(ax_skew, ax_skew)

    s = np.sin(t*angle)*ax_skew
    c = (1-np.cos(t*angle))*ax_skew2

    Rt = np.matmul(R0, np.eye(3) + s + c)
    return Rt

def R_pitch(pitch0, delta_pitch, t):
    pitch = (1-t)*pitch0 + t*delta_pitch
    c = np.cos(pitch)
    s = np.sin(pitch)
    R = np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R

# def R_avg(R_list):
#
#     # R_avg = product_{i=1}^{N} R_i^(1/N) = e^(1/N * product_{i=1}^{N} R_i)
#     N = len(R_list)
#     Logs = np.eye(3)
#     for R in R_list:
#         Logs = Logs @ l.logm(R)
#     avg = l.expm(1/N * Logs)
#     return avg

def R_avg_list(R_list):
    # only if len(R_list) = 4
    # avg(R1, R2) = R1 * sqrt(R1^T * R2)
    # see MAHER MOAKHER - MEANS AND AVERAGING IN THE GROUP OF ROTATIONS

    R01 = R_avg(R_list[0], R_list[1])
    R23 = R_avg(R_list[2], R_list[3])

    return R_avg(R01, R23)


def R_avg(R0, R1):
    return (R0 @ l.sqrtm(R0.transpose() @ R1)).real




