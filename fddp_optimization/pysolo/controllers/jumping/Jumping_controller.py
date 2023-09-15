import numpy as np
from pysolo.controllers.jumping.Lander import Lander
from pysolo.controllers.jumping.Thruster import Thruster


# Pinocchio modules

#np.set_printoptions(suppress=True, precision=3, linewidth=os.get_terminal_size(0)[0])

class JumpingController:
    def __init__(self, robot, x_max=0., y_max=0., z_max=0., tau_max=11.25, T_landing = 3.0, dt=0.001,
                 vectors_for_plot = True):
        '''
        Args:
            robot: TSID model
            x_max: range, max com_x (x_max > q_home[0])
            z_max: apex, max com_z (z_max > q_home[2])
            tau_max: maximum achievable torque [tau_max = max current * torque constant]
            T_landing: time interval dedicated for landing controller
                       the total time horizon is T_stance+T_air+T_landing
            dt: sampling time
        '''

        # INITIALIZATION CONTROLLERS
        self.vectors_for_plots = vectors_for_plot
        self.thruster = Thruster(robot, x_max=x_max, y_max=y_max, z_max=z_max, T_landing = T_landing,
                                 vectors_for_plots = self.vectors_for_plots)

        self.T_stance = self.thruster.T_stance
        self.T_air = self.thruster.T_air
        T_total = self.thruster.T_total
        sample_at_lift_off = self.thruster.sample_at_lift_off
        sample_at_touch_down = self.thruster.sample_at_touch_down

        self.lander = Lander(robot, T_total=T_total)


        self.total_samples = self.thruster.samples
        self.sample_td = 0                          # sample at touch down
        self.real_td = 0
        self.time_span = self.thruster.time_span

        self.touched = False
        self.lift_off = False


        if self.vectors_for_plots:
            self.com_plot = np.zeros([self.total_samples, 3])
            self.comdot_plot = np.zeros([self.total_samples, 3])
            self.com_ref_plot = np.zeros([self.total_samples, 3])
            self.comdot_ref_plot = np.zeros([self.total_samples, 3])
            self.force_plot = np.zeros([self.total_samples, 4, 3])
            self.force_fb_plot = np.zeros([self.total_samples, 4, 3])
            self.force_ff_plot = np.zeros([self.total_samples, 4, 3])
            self.torque_plot = np.zeros([self.total_samples, 12])
            self.torque_fb_plot = np.zeros([self.total_samples, 12])
            self.torque_ff_plot = np.zeros([self.total_samples, 12])
            self.qjoint_plot = np.zeros([self.total_samples, 12])
            self.vjoint_plot = np.zeros([self.total_samples, 12])
            self.posbase_plot = np.zeros([self.total_samples, 7])
            self.velbase_plot = np.zeros([self.total_samples, 6])



    def compute(self, q, qdot, i, t):
        #data = self.thruster.compute(q, qdot, i, t)

        if self.lift_off == False and self.touched==False:
            print('thrusting phase')
            data = self.thruster.compute(q, qdot, i, t)
            self.lift_off = self.has_lift_off(q, qdot, t)
        elif self.lift_off == True and self.touched==False:
            print('in air')
            data = self.thruster.compute(q, qdot, i, t, self.lift_off)
            self.touched = self.has_touched(q, qdot, i, t)
            self.real_td = t
        elif self.touched == True:
            print('landing')
            data = self.lander.compute(q, qdot, i, t)


        if self.vectors_for_plots:
            self._fill_vectors_for_plots(i, data)

        return data['torque'].flatten() #q, qdot


    def has_touched(self, q, qdot, i, t):
        if t <= 0.1:
            return False
        ret = self.thruster.touch_down(q, qdot, t)      # if True, notifies the touch down
        if ret:
            self.sample_td = i
            self.lander.touch_down(q, qdot, t)          # run initialization of the lander controller
        return ret


    def has_lift_off(self, q, qdot, t):
        if t <= 0.1:
            return False
        ret = self.thruster.lift_off(q, qdot, t)        # if True, notifies the lift off
        return ret


    def _fill_vectors_for_plots(self, sample, data):
        # in universe frame: com, comdot, com_ref, comdot_ref, force, force_fb, force_ff, torque,
        # torque_fb, torque_ff
        q = data['q']
        qdot = data['qdot']
        com = data['com']
        comdot = data['comdot']

        self.com_plot[sample, :] = com
        self.comdot_plot[sample, :] = comdot
        self.qjoint_plot[sample, :] = q[7:]
        self.vjoint_plot[sample, :] = qdot[6:]
        self.posbase_plot[sample, :] = q[:7]
        self.velbase_plot[sample, :] = qdot[:6]
        self.com_ref_plot[sample, :] = data['com_ref']
        self.comdot_ref_plot[sample, :] = data['comdot_ref']

        force = data['force']
        force_fb = data['force_fb']
        force_ff = data['force_ff']
        for i in range(0,4):
            self.force_plot[sample, i, :] = force[3*i: 3*(i+1)]
            self.force_fb_plot[sample, i, :]  = force_fb[3*i: 3*(i+1)]
            self.force_ff_plot[sample, i, :]  = force_ff[3*i: 3*(i+1)]

        self.torque_plot[sample, :] = data['torque']
        self.torque_fb_plot[sample, :] = data['torque_fb']
        self.torque_ff_plot[sample, :] = data['torque_ff']
