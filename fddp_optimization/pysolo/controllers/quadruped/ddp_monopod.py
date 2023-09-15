import crocoddyl
import numpy as np
import pinocchio
import pinocchio as pin


class DDPMonopodRobot:
    def __init__(self, robot, lfFootName, baseName):
        # Robot
        self.robot = robot
        self.rmodel = self.robot.model.copy()
        self.rdata = self.rmodel.createData()
        # Dynamics
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # integrator
        self.integrator = 'euler'  # by default  other runge kutta

        # state bounds (size = state.ndx (= nq-1 + nv, for floating base robots))
        # self.state_lb = np.delete(self.state.lb, 3)
        # self.state_ub = np.delete(self.state.ub, 3)
        self.state_lb =  self.state.lb
        self.state_ub =  self.state.ub
        # actuation bounds (size = state.nq-6)
        self.actuation_lb = -self.rmodel.effortLimit[3:]
        self.actuation_ub = self.rmodel.effortLimit[3:]

        # Getting the frame id for all the feet and for the base
        lfFootId = self.rmodel.getFrameId(lfFootName)
        self.feetId = {'lf': lfFootId}
        self.baseId = self.rmodel.getFrameId(baseName)


        # To Be Overwritten in the problems
        self.rmodel.defaultState = pin.neutral(self.rmodel)
        # suppose that the contact surface has the same friction coefficient everywhere
        # I consider only initial and final orientation of the contact surface
        self.mu = 0.0
        self.RSurf = {'lf': np.full([3,3], np.nan)}


    # contact model (better)
    def createContactModel(self, supportFeet, feetRef, BaumgarteGains):
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for foot in supportFeet:
            idx = supportFeet[foot]
            footRef = feetRef[foot]
            supportContactModel = crocoddyl.ContactModel3D(self.state,
                                                           idx,
                                                           footRef.translation,
                                                           pinocchio.LOCAL_WORLD_ALIGNED,
                                                           self.actuation.nu,
                                                           BaumgarteGains)


            
            contactModel.addContact(self.rmodel.frames[idx].name + "_contact", supportContactModel)
        return contactModel

    def createImpulseContactModel(self, supportFeet):
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for foot in supportFeet:
            idx = supportFeet[foot]
            supportContactModel = crocoddyl.ImpulseModel3D(self.state, idx)
            impulseModel.addImpulse(self.rmodel.frames[idx].name + "_impulse", supportContactModel)
        return impulseModel


      ########    ########      ########  ########   ########
     ##          ##      ##    ##            ##     ##
    ##          ##        ##    #######      ##      #######
     ##          ##      ##           ##     ##            ##
      ########    ########     ########      ##     ########

    def frictionConeCostModel(self, foot_name, nfacets=4, inner_appr=True):
        idx = self.feetId[foot_name]
        frictionCone = crocoddyl.FrictionCone(self.RSurf[foot_name],
                                              self.mu,
                                              nf=nfacets,
                                              inner_appr=inner_appr)
        activationBounds = crocoddyl.ActivationBounds(frictionCone.lb, frictionCone.ub)
        frictionConeActivation = crocoddyl.ActivationModelQuadraticBarrier(activationBounds)
        frictionConeResidual = crocoddyl.ResidualModelContactFrictionCone(self.state,
                                                                          idx,
                                                                          frictionCone,
                                                                          self.actuation.nu)
        frictionConeCost = crocoddyl.CostModelResidual(self.state,
                                                       frictionConeActivation,
                                                       frictionConeResidual)

        return frictionConeCost

    def framePlacementCostModel(self, id, frameRef, framePlacementDirectionPenalties=None, nu=None):
        # frameRef is expressed in World frame (as oMf)
        if nu is None:
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state,
                                                                              id,
                                                                              frameRef,
                                                                              self.actuation.nu)
        else:
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state,
                                                                           id,
                                                                           frameRef,
                                                                           nu)
        if framePlacementDirectionPenalties is None:
            # by default, activation model is quadratic, i.e. a = 1/2 res^T*res
            framePlacementCost = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
        else:
            # weighed quadratic, a = 1/2 res^T*W*res
            framePlacementActivationModel = crocoddyl.ActivationModelWeightedQuad(np.array(framePlacementDirectionPenalties) ** 2)
            framePlacementCost = crocoddyl.CostModelResidual(self.state,
                                                             framePlacementActivationModel,
                                                             framePlacementResidual)
        return framePlacementCost

    def frameVelocityCostModel(self, id, frameVelocity, referenceFrame, baseVelocityPenalty=None):
        frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state,
                                                                     id,
                                                                     frameVelocity,
                                                                     referenceFrame,
                                                                     self.actuation.nu)
        if baseVelocityPenalty is None:
            # by default, activation model is quadratic, i.e. a = 1/2 res^T*res
            framePlacementCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
        else:
            framePlacementActivationModel = crocoddyl.ActivationModelWeightedQuad(np.array(baseVelocityPenalty) ** 2)
            framePlacementCost = crocoddyl.CostModelResidual(self.state,
                                                             framePlacementActivationModel,
                                                             frameVelocityResidual)
        return framePlacementCost

    def comCostModel(self, comRef, comPenalty=None):
        comPlacementResidual = crocoddyl.ResidualModelCoMPosition (self.state,
                                                                   comRef,
                                                                   self.actuation.nu)
        if comPenalty is None:
            # by default, activation model is quadratic, i.e. a = 1/2 res^T*res
            comPlacementCost = crocoddyl.CostModelResidual(self.state, comPlacementResidual)
        else:
            comPlacementActivationModel = crocoddyl.ActivationModelWeightedQuad(np.array(comPenalty) ** 2)
            comPlacementCost = crocoddyl.CostModelResidual(self.state,
                                                             comPlacementActivationModel,
                                                             comPlacementResidual)
        return comPlacementCost

    def stateRegularizationCostModel(self, defaultState, statePenalty=None, nu=None):
        if nu is None:
            stateRegResidual = crocoddyl.ResidualModelState(self.state, defaultState, self.actuation.nu)
        else:
            stateRegResidual = crocoddyl.ResidualModelState(self.state, defaultState, 0)
        if statePenalty is None:
            stateRegCost = crocoddyl.CostModelResidual(self.state,
                                                       stateRegResidual)
        else:
            stateRegActivationModel = crocoddyl.ActivationModelWeightedQuad(np.array(statePenalty) ** 2)
            stateRegCost = crocoddyl.CostModelResidual(self.state,
                                                       stateRegActivationModel,
                                                       stateRegResidual)
        return stateRegCost

    def stateBoundsCostModel(self, state_lb, state_ub):
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, self.actuation.nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(state_lb, state_ub))
        stateBoundsCost = crocoddyl.CostModelResidual(self.state,
                                                      stateBoundsActivation,
                                                      stateBoundsResidual)
        return stateBoundsCost

    def actuationRegularizationCostModel(self, actuationWeights=None):
        actuationRegResidual = crocoddyl.ResidualModelControl(self.state, self.actuation.nu)
        if actuationWeights is None:
            actuationRegCost = crocoddyl.CostModelResidual(self.state,
                                                           actuationRegResidual)
        else:
            actuationRegActivationModel = crocoddyl.ActivationModelWeightedQuad(np.array(actuationWeights) ** 2)
            actuationRegCost = crocoddyl.CostModelResidual(self.state,
                                                           actuationRegActivationModel,
                                                           actuationRegResidual)
        return actuationRegCost

    def actuationBoundsCostModel(self, actuation_lb, actuation_ub):
        actuationRegResidual = crocoddyl.ResidualModelControl(self.state, self.actuation.nu)
        actuationActivationBounds = crocoddyl.ActivationBounds(actuation_lb, actuation_ub)
        actuationBoundsActivationModel = crocoddyl.ActivationModelQuadraticBarrier(actuationActivationBounds)
        actuationBoundsCost = crocoddyl.CostModelResidual(self.state,
                                                          actuationBoundsActivationModel,
                                                          actuationRegResidual)

        return actuationBoundsCost

    ########     #####      ########  ##    ##   ########
       ##       ##   ##    ##         ##   ##   ##
       ##      #########    #######   ######     #######
       ##     ##       ##         ##  ##   ##          ##
       ##     ##       ##  ########   ##    ##  ########

    def createBasePushUpProblem(self, x0, problemDescription):
        # PushUp problem consists in periodic base height variation
        # z(k) = z_0 + amplidude * sin(2*pi*frequency*timeStep*k)

        # problem description parameters
        timeStep = problemDescription['timeStep']
        frequency = problemDescription['frequency']
        amplitude = problemDescription['amplitude']
        n_cycles = problemDescription['n_cycles']
        for foot in problemDescription['rpySurf']:
            self.RSurf[foot] = pin.rpy.rpyToMatrix(np.array(problemDescription['rpySurf'][foot]))
        self.mu = problemDescription['mu']
        weights = problemDescription['weights']

        self.integrator = problemDescription['integrator']

        totalKnots = int(n_cycles / (timeStep * frequency))

        self.rmodel.defaultState = x0.copy()
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        basePose_init = self.rdata.oMf[self.baseId].copy()
        delta_basePose = pin.SE3(np.eye(3), np.zeros(3))

        feetPoseInit = {}
        for foot in self.feetId:
            feetPoseInit[foot] = self.rdata.oMf[self.feetId[foot]].copy()

        loco3dModel = []

        for k in range(totalKnots):
            delta_z = np.array(amplitude) * np.sin(2 * np.pi * frequency * k * timeStep)
            delta_basePose.translation = np.array([0., 0., delta_z])

            model = self.createContactsAndBaseActionModel(timeStep=timeStep,
                                                          supportFeet=self.feetId,
                                                          feetRef=feetPoseInit,
                                                          basePoseRef=basePose_init.act(delta_basePose),
                                                          weights=weights)
            loco3dModel.append(model)

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem

    def createBaseWobblingProblem(self, x0, problemDescription):
        # Wobbling problem consists in periodic base rotation in RPY (not necessarily decoupled)
        # RPY(k) = RPY_0 + amplidudes * sin(2*pi*frequency*timeStep*k)

        # problem description parameters
        timeStep = problemDescription['timeStep']
        frequency = problemDescription['frequency']
        amplitudes = problemDescription['amplitudes']
        n_cycles = problemDescription['n_cycles']
        for foot in problemDescription['rpySurf']:
            self.RSurf[foot] = pin.rpy.rpyToMatrix(np.array(problemDescription['rpySurf'][foot]))
        self.mu = problemDescription['mu']
        weights = problemDescription['weights']
        self.integrator = problemDescription['integrator']
        totalKnots = int(n_cycles /(timeStep*frequency))

        self.rmodel.defaultState = x0
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        basePose_init = self.rdata.oMf[self.baseId].copy()
        delta_basePose = pin.Pose(np.eye(3), np.zeros(3))

        feetPoseInit = {}
        for foot in self.feetId:
            feetPoseInit[foot] = self.rdata.oMf[self.feetId[foot]].copy()


        loco3dModel = []

        for k in range(totalKnots):
            delta_rpy = np.array(amplitudes) * np.sin(2*np.pi*frequency * k * timeStep)
            delta_basePose.rotation = pin.rpy.rpyToMatrix(delta_rpy)

            model = self.createContactsAndBaseActionModel(timeStep=timeStep,
                                                          supportFeet=self.feetId,
                                                          feetRef=feetPoseInit,
                                                          basePoseRef=basePose_init.act(delta_basePose),
                                                          weights=weights)
            loco3dModel.append(model)

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem

    def createJumpOnHorizontalTerrainProblem(self, x0, problemDescription):
        # this task consists of a jump on horizontal surface

        # temporal description
        timeStep = problemDescription['timeStep']

        launchingT = problemDescription['launchingT']
        # nodi in salita
        launchingKnots = int(launchingT/timeStep)

        landedT  = problemDescription['landedT']
        landedKnots = int(landedT/timeStep)
        # geometric description
        jumpHeight = problemDescription['jumpHeight']
        jumpLength = np.array(problemDescription['jumpLength'])

        for foot in problemDescription['rpySurf']:
            self.RSurf[foot] = pin.rpy.rpyToMatrix(np.array(problemDescription['rpySurf'][foot]))

        finalYaw = problemDescription['finalYaw']
        self.mu = problemDescription['mu']

        weights = problemDescription['weights']
        self.integrator = problemDescription['integrator']

        # Compute initial kinematics
        self.rmodel.defaultState = x0.copy()
        q0 = self.rmodel.defaultState[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        feetPoseInit = {}
        for foot in self.feetId:
            feetPoseInit[foot] = self.rdata.oMf[self.feetId[foot]].copy()
            #print("foot pos init: ", feetPoseInit[foot])
        basePoseInit = self.rdata.oMf[self.baseId].copy()
        #print("base pos init: ", basePoseInit)

        # Compute initial COM
        comOffset = pin.centerOfMass(self.rmodel, self.rdata, q0) - basePoseInit.translation
        comInit = pin.centerOfMass(self.rmodel, self.rdata, q0)

        # Divide the jumping problem in different phases
        loco3dModel = []

        ### thrusting PHASE ###
        takeOffPhase = []
        for k in range(launchingKnots+1):
            model = self.createContactsAndBaseActionModel(timeStep=timeStep,
                                                          supportFeet=self.feetId,
                                                          feetRef=feetPoseInit,
                                                          basePoseRef=basePoseInit,
                                                          weights=weights['thrustingPhase'])
            takeOffPhase.append(model)

        ### FLYING UP PHASE ###
        flyingUpPhase = []

        # only for jumps flat terrain
        #comApex = comInit + jumpLength / 2  + np.array([0., 0., jumpHeight])
        #for jumps at higher levels
        r_line = jumpLength/np.linalg.norm(jumpLength)
        y_axis = np.cross(r_line, np.array([0,0,1]))
        norm_line = np.cross(y_axis, r_line)

        comApex = comInit + r_line*(np.linalg.norm(jumpLength)/2) + jumpHeight*norm_line

        if np.linalg.norm(jumpLength[:2]) == 0:
            norm_line = np.array([0,0,1])
            comApex = comInit + norm_line * (jumpLength[2] + jumpHeight)

        # print("comInit: ", comInit)
        # print("comApex: ", comApex)

        self.footRef = []
        self.comRef= []
        self.basePoseRef = []
        delta_z = comApex[2]-comInit[2]


        flyingUpT = np.sqrt(2*delta_z/(-self.rmodel.gravity.linear[2]))#+0.1 for backward jump
        print("flying time: ", 2*flyingUpT)
        flyingUpKnots = int(flyingUpT/timeStep)

        basePoseApex = basePoseInit.copy()
        basePoseApex.translation = comApex - comOffset
        basePoseApex.rotation = pin.rpy.rpyToMatrix(0., 0., finalYaw/2)

        feetPoseRef = {}
        for k in range(flyingUpKnots+1):
            alpha = k/flyingUpKnots
            comRef = alpha*comApex + (1-alpha)*comInit
            basePoseRef = pin.SE3.Interpolate(basePoseInit, basePoseApex, alpha)
            self.comRef.append(comRef)
            self.basePoseRef.append(pin.rpy.matrixToRpy(basePoseRef.rotation))

            for foot in feetPoseInit:
                feetPoseRef[foot] = basePoseInit.actInv(basePoseRef).act(feetPoseInit[foot])
            self.footRef.append(feetPoseRef['lf'].translation)
            model = self.createFlyingActionModel(timeStep=timeStep,
                                                 comRef=comRef,
                                                 feetRef=feetPoseRef,
                                                 basePoseRef=basePoseRef,
                                                 weights=weights['flyingPhase'])
            flyingUpPhase.append(model)

        ### FLYING DOWN PHASE ###
        flyingDownPhase = []

        comFinal = comInit + jumpLength

        #print("comFinal: ", comFinal)
        # this can only be used for flat terrains where cannot happen than the Comapex is lower  than the comFinal
        #delta_z = comApex[2]-comFinal[2]
        #flyingDownT = np.sqrt(2 * delta_z / (-self.rmodel.gravity.linear[2]))
        flyingDownT = flyingUpT
        flyingDownKnots = int( flyingDownT / timeStep )
        # because of the mask in weights['flyingPhase'][basePlacementDirectionScale], there is no problem if one add padding in other SE3
        # components.
        basePoseFinal = basePoseApex.copy()
        basePoseFinal.translation = comFinal -comOffset
        basePoseFinal.rotation = pin.rpy.rpyToMatrix(0., 0., finalYaw) # use only angular part

        #print("comFinal: ", comFinal)
        #print("base pos final: ", basePoseFinal)

        feetPoseRef = {}
        for k in range(1, flyingDownKnots+1):
            alpha = k / flyingDownKnots
            comRef = alpha * comFinal + (1 - alpha) * comApex
            basePoseRef = pin.SE3.Interpolate(basePoseApex, basePoseFinal, alpha)
            for foot in feetPoseInit:
                feetPoseRef[foot] = basePoseInit.actInv(basePoseRef).act(feetPoseInit[foot])


            self.footRef.append(feetPoseRef['lf'].translation)
            self.comRef.append(comRef)
            self.basePoseRef.append(pin.rpy.matrixToRpy(basePoseRef.rotation))
            model = self.createFlyingActionModel(timeStep=timeStep,
                                                 feetRef=feetPoseRef,
                                                 comRef=comRef,
                                                 basePoseRef=basePoseRef,
                                                 weights=weights['flyingPhase'])
            flyingDownPhase.append(model)



        # ### LANDING PHASE ###
        feetPoseFinal = {}
        for foot in feetPoseInit:
            feetPoseFinal[foot] = feetPoseRef[foot].copy()
            #print("feetPose  final: ", feetPoseRef[foot])
        # DEBUG
        # print('basePoseInit', basePoseInit)
        # print('basePoseApex', basePoseApex)
        # print('basePoseFinal', basePoseFinal)
        #
        # for foot in feetPoseFinal:
        #     print('feetPoseInit', foot, feetPoseInit[foot])
        #     print('feetPoseFinal', foot, feetPoseFinal[foot])

        landingPhase = []
        model = self.createFootSwitchModel(supportFeet=self.feetId,
                                           feetRef=feetPoseFinal,
                                           weights=weights['landingPhase'],
                                           pseudoImpulse=False)
        landingPhase.append(model)


        ### LANDED PHASE ###
        landedPhase = []
        for k in range(landedKnots + 1):
            model = self.createContactsAndBaseActionModel(timeStep=timeStep,
                                                          supportFeet=self.feetId,
                                                          feetRef=feetPoseFinal,
                                                          comRef=comFinal,
                                                          basePoseRef=basePoseFinal,
                                                          weights=weights['landedPhase'])
            landedPhase.append(model)

        loco3dModel += takeOffPhase
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landedPhase

        problem = crocoddyl.ShootingProblem(self.rmodel.defaultState, loco3dModel[:-1], loco3dModel[-1])
        return problem




    # common action models
    def createIntegratedActionModel(self, dmodel, timeStep):
        if self.integrator == 'rk4':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.four, timeStep)
        elif self.integrator == 'rk3':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.three, timeStep)
        elif self.integrator == 'rk2':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, timeStep)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model



    def createContactsAndBaseActionModel(self, timeStep, supportFeet=None, feetRef=None, comRef = None, basePoseRef=None, weights=None):
        """ Action model for feet and/or base motions
            Feet and base are treated as end effector
            used by: createBasePushUpProblem, createBaseWobblingProblem, createJumpingOnSlopedTerrain
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        contactModel = self.createContactModel(supportFeet, feetRef, np.array(weights['BaumgarteGains']))

        # Creating the cost model
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)

        ##################
        # Friction Cones #
        ##################
        for foot in supportFeet:
            frictionConeCost = self.frictionConeCostModel(foot)
            costModel.addCost(self.rmodel.frames[self.feetId[foot]].name + "_frictionCone",
                              frictionConeCost,
                              weights['frictionCone'])
        ##################
        # com taqsk  #
        ##################
        if comRef is not None:
            comCost = self.comCostModel(comRef)
            costModel.addCost("com",
                              comCost,
                              weights['com'])

        #############
        # Feet task #
        #############
        if feetRef is not None:
            for foot in feetRef:
                id = self.feetId[foot]
                footRef = feetRef[foot]

                footPlacementCost = self.framePlacementCostModel(id, footRef)
                costModel.addCost(foot + "_footTrack",
                                  footPlacementCost,
                                  weights['footPlacement'])

                # footVelocityCost = self.frameVelocityCostModel(id, pin.Motion.Zero(), pin.LOCAL)
                # costModel.addCost(foot + "_impulseVel",
                #                   footVelocityCost,
                #                   weights['footVelocity'])

        #######################
        # Base Placement Task #
        #######################
        if basePoseRef is not None:
            basePlacementCost = self.framePlacementCostModel(self.baseId, basePoseRef, weights['basePlacementDirectionPenalty'])
            costModel.addCost(self.rmodel.frames[self.baseId].name + "_PlacementCost",
                              basePlacementCost,
                              weights['basePlacement'])


        ########################
        # State Regularization #
        ########################
        stateRegCost = self.stateRegularizationCostModel(self.rmodel.defaultState, np.array(weights['statePenalty']))
        costModel.addCost("stateReg",
                          stateRegCost,
                          weights['stateRegularization'])

        ################
        # State Bounds #
        ################
        stateBoundsCost = self.stateBoundsCostModel(self.state_lb,  self.state_ub)
        costModel.addCost("stateBounds",
                          stateBoundsCost,
                          weights['stateBounds'])

        ############################
        # Actuation Regularization #
        ############################
        actuationRegCost = self.actuationRegularizationCostModel()
        costModel.addCost("actuationReg",
                          actuationRegCost,
                          weights['actuationRegularization'])

        ####################
        # Actuation Bounds #
        ####################
        actuationBoundsCost = self.actuationBoundsCostModel(self.actuation_lb, self.actuation_ub)
        costModel.addCost("actuationBounds",
                          actuationBoundsCost,
                          weights['actuationBounds'])

        ######################
        # Differential Model #
        ######################
        # Creating the action model for the KKT dynamics
        if len(supportFeet) == 0:
            enable_force = False
            JMinvJt_damping = 0.
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                         costModel, JMinvJt_damping, enable_force)
        else:
            enable_force = True
            JMinvJt_damping = 0.
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                         costModel, JMinvJt_damping, enable_force)

        model = self.createIntegratedActionModel(dmodel, timeStep)
        return model

    def createFlyingActionModel(self, timeStep, feetRef, comRef, basePoseRef, weights):
        """ Action model for feet not in contact and com motion, use basePoseRef only for (yaw) rotations
            used by:
        """

        # Creating the cost model
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)

        #############
        # Feet task #
        #############
        if feetRef is not None:
            for foot in feetRef:
                id = self.feetId[foot]
                footRef = feetRef[foot]

                footPlacementCost = self.framePlacementCostModel(id, footRef)
                costModel.addCost(foot + "_footTrack",
                                  footPlacementCost,
                                  weights['footPlacement'])

        ############
        # CoM Task #
        ############
        if comRef is not None:
            comCost = self.comCostModel(comRef)
            costModel.addCost("com",
                              comCost,
                              weights['com'])

        ######################
        # Base Rotation Task #
        ######################
        # this task is specifically thought only for yaw rotations
        if basePoseRef is not None:
            baseYawCost = self.framePlacementCostModel(self.baseId, basePoseRef, weights['basePlacementDirectionPenalty'])
            costModel.addCost("baseYaw",
                              baseYawCost,
                              weights['baseYaw'])


        ########################
        # State Regularization #
        ########################
        stateRegCost = self.stateRegularizationCostModel(self.rmodel.defaultState, np.array(weights['statePenalty']))
        costModel.addCost("stateReg",
                          stateRegCost,
                          weights['stateRegularization'])

        ################
        # State Bounds #
        ################
        stateBoundsCost = self.stateBoundsCostModel(self.state_lb,  self.state_ub)
        costModel.addCost("stateBounds",
                          stateBoundsCost,
                          weights['stateBounds'])

        ############################
        # Actuation Regularization #
        ############################
        actuationRegCost = self.actuationRegularizationCostModel()
        costModel.addCost("actuationReg",
                          actuationRegCost,
                          weights['actuationRegularization'])

        ####################
        # Actuation Bounds #
        ####################
        actuationBoundsCost = self.actuationBoundsCostModel(self.actuation_lb, self.actuation_ub)
        costModel.addCost("actuationBounds",
                          actuationBoundsCost,
                          weights['actuationBounds'])

        ######################
        # Differential Model #
        ######################
        # Creating the action model for the free dynamics with simpletic Euler integration scheme
        dmodel = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state,
                                                                  self.actuation,
                                                                  costModel)

        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createFootSwitchModel(self, supportFeet, feetRef, weights, pseudoImpulse=False):
        """ Action model for a foot switch phase.
        :param supportFeet: Ids of the constrained feet
        :param feetRef: reference for the feet foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFeet, feetRef, weights)
        else:
            return self.createImpulseModel(supportFeet, feetRef, weights)

    def createPseudoImpulseModel(self, supportFeet, feetRef, weights):
        """ Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
        :param supportFeet: Ids of the constrained feet
        :param feetRef: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        contactModel = self.createContactModel(supportFeet, feetRef, np.array(weights['BaumgarteGains']))

        # Creating the cost model
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)

        ##################
        # Friction Cones #
        ##################
        for foot in supportFeet:
            frictionConeCost = self.frictionConeCostModel(foot)
            costModel.addCost(self.rmodel.frames[self.feetId[foot]].name + "_frictionCone",
                              frictionConeCost,
                              weights['frictionCone'])

        ######################
        # Swinging foot task #
        ######################
        for foot in feetRef:
            id = self.feetId[foot]
            footRef = feetRef[foot]

            footPlacementCost = self.framePlacementCostModel(id, footRef)
            costModel.addCost(foot + "_footTrack",
                              footPlacementCost,
                              weights['footPlacement'])

            footVelocityCost = self.frameVelocityCostModel(id, pin.Motion.Zero(), pin.LOCAL)
            costModel.addCost(foot + "_impulseVel",
                              footVelocityCost,
                              weights['footVelocity'])

        ########################
        # State Regularization #
        ########################
        stateRegCost = self.stateRegularizationCostModel(self.rmodel.defaultState, np.array(weights['statePenalty']))
        costModel.addCost("stateReg",
                          stateRegCost,
                          weights['stateRegularization'])

        ################
        # State Bounds #
        ################
        stateBoundsCost = self.stateBoundsCostModel(self.state_lb, self.state_ub)
        costModel.addCost("stateBounds",
                          stateBoundsCost,
                          weights['stateBounds'])

        ############################
        # Actuation Regularization #
        ############################
        actuationRegCost = self.actuationRegularizationCostModel()
        costModel.addCost("actuationReg",
                          actuationRegCost,
                          weights['actuationRegularization'])

        ####################
        # Actuation Bounds #
        ####################
        actuationBoundsCost = self.actuationBoundsCostModel(self.actuation_lb, self.actuation_ub)
        costModel.addCost("actuationBounds",
                          actuationBoundsCost,
                          weights['actuationBounds'])

        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        JMinvJt_damping = 0.
        enable_force = True
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state,
                                                                     self.actuation,
                                                                     contactModel,
                                                                     costModel,
                                                                     JMinvJt_damping,
                                                                     enable_force)

        model = self.createIntegratedActionModel(dmodel, 0.)
        return model

    def createImpulseModel(self, supportFeet, feetRef, weights, JMinvJt_damping=1e-12, r_coeff=0.0):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = self.createImpulseContactModel(supportFeet)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        ######################
        # Swinging foot task #
        ######################
        for foot in feetRef:
            id = self.feetId[foot]
            footRef = feetRef[foot]

            footPlacementCost = self.framePlacementCostModel(id, footRef, nu=0)
            costModel.addCost(foot + "_footTrack",
                              footPlacementCost,
                              weights['footPlacement'])

        ########################
        # State Regularization #
        ########################
        stateRegCost = self.stateRegularizationCostModel(self.rmodel.defaultState, np.array(weights['statePenalty']), 0)
        costModel.addCost("stateReg",
                          stateRegCost,
                          weights['stateRegularization'])



        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model




    def generateProblem(self, x0, problemDescription):
        # define the problem
        if problemDescription['type'] == 'wobbling':
            prob = self.createBaseWobblingProblem(x0,problemDescription)
        elif problemDescription['type'] == 'pushup':
            prob = self.createBasePushUpProblem(x0, problemDescription)
        elif problemDescription['type'] == 'jumpOnHorizontalTerrain':
            prob = self.createJumpOnHorizontalTerrainProblem(x0, problemDescription)
        else:
            assert False, "problem description type is unknown"



        # choose the solver
        if problemDescription['solver'] == 'DDP':
            ddp_prob = crocoddyl.SolverDDP(prob)
        elif problemDescription['solver'] == 'FDDP':
            ddp_prob = crocoddyl.SolverFDDP(prob)
        elif problemDescription['solver'] == 'BoxDDP':
            ddp_prob = crocoddyl.SolverBoxDDP(prob)
        elif problemDescription['solver'] == 'BoxFDDP':
            ddp_prob = crocoddyl.SolverBoxFDDP(prob)
        else:
            assert False, "problem solver type is unknown"

        # set the callbacks
        cb_list = []
        if problemDescription['callbacks']['logger']:
            cb_list.append(crocoddyl.CallbackLogger())
        if problemDescription['callbacks']['verbose']:
            cb_list.append(crocoddyl.CallbackVerbose())
        ddp_prob.setCallbacks(cb_list)

        return ddp_prob


    def analyzeSolution(self, ddp_solver, verbose=True):
        costs_model = []
        costs_data = []

        for i in range(ddp_solver.problem.T):
            model = ddp_solver.problem.runningModels[i]
            if 'ImpulseFwdDynamics' in str(type(model)):
                costs_model.append(model.costs.costs)
            else:
                costs_model.append(model.differential.costs.costs)

            data = ddp_solver.problem.runningDatas[i]
            if 'ImpulseFwdDynamics' in str(type(data)):
                costs_data.append(data.costs.costs)
            else:
                costs_data.append(data.differential.costs.costs)

        costs_model.append(ddp_solver.problem.terminalModel.differential.costs.costs)
        costs_data.append(ddp_solver.problem.terminalData.differential.costs.costs)

        dictCosts = {}
        dictWCosts = {}

        cumSum = 0.
        cumWSum = 0.
        for i in range(ddp_solver.problem.T):
            cm = costs_model[i]
            cd = costs_data[i]

            keys = list(cm.todict().keys())
            for k in keys:
                dictCosts[k] = cd[k].cost + dictCosts.get(k, 0.)
                cumSum += cd[k].cost
                wCost = cm[k].weight*cd[k].cost
                dictWCosts[k] = wCost + dictWCosts.get(k, 0.)
                cumWSum += wCost
        dictCosts['cumulative'] = cumSum
        dictWCosts['cumulative'] = cumWSum

        if verbose:
            print('*** Unweighted costs ***')
            for key in dictCosts:
                print(key, dictCosts[key])


            print('\n*** Weighted costs ***')
            for key in dictWCosts:
                print(key, dictWCosts[key])

        return dictCosts, dictWCosts