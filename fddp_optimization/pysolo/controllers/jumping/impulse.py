import numpy as np

class BezierPoly:
    def __init__(self, ctrl_pts):
        # control points
        self.n = ctrl_pts.shape[0]
        self.bezier_coeffs = []
        for i in range(self.n):
            self.bezier_coeffs.append(np.math.comb(self.n, i)* ctrl_pts[i])

        self.poly_coeffs = np.zeros(self.n)
        for i in range(self.n):
            for k in range(self.n-i):
                tmp = np.math.comb(self.n-i-1, k) * (-1)**k # binomial theorem
                self.poly_coeffs[k+i] += np.math.comb(self.n-1, i) * tmp * ctrl_pts[i]

    def compute_using_bezier_formula(self, t): # deprecated
        res = 0
        if t < 0. or t > 1.:
            return res
        for i in range(0, self.n):
            res += self.bezier_coeffs[i] * t**i * (1-t)**(self.n-i)
        return res

    def compute(self, t):
        res = 0
        if t < 0. or t > 1.:
            return res
        for i in range(self.n):
            res += self.poly_coeffs[i] * t**i
        return res

    def integrate(self, t_min, t_max):
        if t_min < 0.:
            t_min = 0.
        if t_max > 1.:
            t_max = 1.

        res = 0.
        if t_max < t_min:
            return res

        for i in range(self.n):
            res += self.poly_coeffs[i] * (t_max**(i+1) - t_min**(i+1))/(i+1)

        return res



class ImpulseThruster:
    def __init__(self, delta_z, delta_x, f_max, timeStart, bezier_coeffs, z0, zmin, robot_mass, g=9.81, s_peak=0.5):
        avg = np.average(bezier_coeffs)
        # self.T_air = np.sqrt(8 * delta_z / g)
        # self.T_stance = (robot_mass * g * self.T_air) / (2 * f_max * avg - robot_mass * g)

        self.T_air = np.sqrt(8/3 * delta_z/g)
        self.T_stance = (robot_mass * g * self.T_air) / (2 * f_max * avg - robot_mass * g)



        self.timesF = {'min': timeStart['F'], 'max': timeStart['F'] + self.T_stance}
        self.timesH = {'min': timeStart['H'], 'max': timeStart['H'] + self.T_stance}
        self.bezier_poly_up = BezierPoly(bezier_coeffs[0])
        self.bezier_poly_down = BezierPoly(bezier_coeffs[1])

        self.alpha_z = f_max
        self.alpha_x = delta_x / (2* avg * self.T_air * self.T_stance)
        self.s_peak = s_peak


        A = np.array([[1, 0, 0, 0, 0], # init pos
                      [0, 1, 0, 0, 0], # init vel
                      [1, self.s_peak, self.s_peak**2, self.s_peak**3, self.s_peak**4], # peak pos
                      [0, 1, 2*self.s_peak, 3*self.s_peak**2, 4*self.s_peak**3], # peak vel
                      [1, 1, 1, 1, 1]]) # final pos

        b =  np.array([[z0],
                      [0],
                      [zmin],
                      [0],
                      [z0]])

        self.z_coeffs = np.linalg.inv(A)@b


    def stanceAchievement(self, t, t_min, t_max):
        if t < t_min:
            s = 0.
        elif t_min <= t <= t_max:
            s = (t - t_min) / (t_max - t_min)
        else:
            s = 1.
        return s

    def force_reference(self, s):
        if s == 0:
            sigma = 0.
            f_ref = 0.
        elif 0 < s <= self.s_peak:
            # compute sigma
            # sigma = 0 for s = 0 (t = t_min)
            # sigma = 0.5 for s = s_peak (t = s_peak*(t_max-t_min) + t_min)
            sigma = s/self.s_peak
            f_ref = self.bezier_poly_up.compute(sigma)
        elif self.s_peak < s < 1:
            # compute sigma
            # sigma = 0.5 for s = s_peak (t = s_peak*(t_max-t_min) + t_min)
            # sigma = 1 for s = 1 (t = t_max)
            sigma = (s-self.s_peak) / (1 - self.s_peak)
            f_ref = self.bezier_poly_down.compute(sigma)
        else:
            sigma = 1.
            f_ref = 0.
        return f_ref, sigma

    def z_reference(self, s):
        z_ref = 0.
        zdot_ref = 0.
        for i in range(4):
            z_ref += self.z_coeffs[i] * s**i
        for i in range(1, 4):
            zdot_ref += i*self.z_coeffs[i] * s**(i-1)
        return z_ref, zdot_ref




    def ffwd(self, t):
        ####################
        # FRONT FFWD FORCE #
        ####################
        # compute s: percentage of achievement of the stance phase
        # s = 0 for t = t_min
        # s = 1 for t = t_max
        s_F = self.stanceAchievement(t, self.timesF['min'], self.timesF['max'])
        s_H = self.stanceAchievement(t, self.timesH['min'], self.timesH['max'])
        fref_F, sigma_F = self.force_reference(s_F)

        F_xF = - self.alpha_x * fref_F
        F_zF = self.alpha_z * fref_F

        fref_H, sigma_H = self.force_reference(s_H)

        F_xH = - self.alpha_x * fref_H
        F_zH = self.alpha_z * fref_H

        zF_ref, zdotF_ref = self.z_reference(s_F)
        zH_ref, zdotH_ref = self.z_reference(s_H)

        z_ref = 0.5 * (zF_ref+ zH_ref)
        zdot_ref = 0.5 * (zdotF_ref+ zdotH_ref)

        return F_xF, F_zF, F_xH, F_zH, z_ref, zdot_ref