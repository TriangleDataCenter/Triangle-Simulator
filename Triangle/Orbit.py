import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import cumulative_trapezoid

from Triangle.Constants import *

class Orbit():
    """
    return dictionaries of functions.
    all the functions take time array (N) as input and return (N, dim) array
    the start time of TCB is set to t = 0, and TCB/TPS coincide at t = 0
    """
    def __init__(self, OrbitDir, max_rows = None, tstart = 0, dt = DAY, pn_order=2):
        # read in orbit data
        # each item is a N * 3 array
        if max_rows == None:
            self.rdata = {key: np.loadtxt(OrbitDir + '/SCP' + key + '.dat') * AU for key in SC_labels}
            self.vdata = {key: np.loadtxt(OrbitDir + '/SCV' + key + '.dat') * AU / DAY for key in SC_labels}
        else:
            self.rdata = {key: np.loadtxt(OrbitDir + '/SCP' + key + '.dat', max_rows=max_rows) * AU for key in SC_labels}
            self.vdata = {key: np.loadtxt(OrbitDir + '/SCV' + key + '.dat', max_rows=max_rows) * AU / DAY for key in SC_labels}
        
        self.tstart = tstart
        self.N = len(self.rdata[SC_labels[0]])
        self.tdata = np.arange(self.N) * dt - self.tstart
        self.dt = dt
        
        # calculate LTT, arm vector
        LTTdata = {}
        self._LTTfunctions = {}
        self._Dopplerfunctions = {}

        ARMdata = {}
        self._ArmVectorfunctions = {}

        self._Positionfunctions = {}
        self._Velocityfunctions = {} 

        for label in MOSA_labels:
            send_SC = label[1]
            receive_SC = label[0]

            L = self.rdata[receive_SC] - self.rdata[send_SC]
            # 0-th order
            LTT0 = np.sqrt(np.sum(L * L, axis=1)) / C
            ARM0 = L / LTT0[:, np.newaxis] / C
            # 1/2-th order
            nv_recv = np.sum(ARM0 * self.vdata[receive_SC], axis=1)
            LTT1 = np.sum(L * self.vdata[receive_SC], axis=1) / C ** 2
            ARM1 = self.vdata[receive_SC] / C - ARM0 * nv_recv[:, np.newaxis] / C
            # 1-st order
            LTT2 = 0.5 * (np.sum((self.vdata[receive_SC]) ** 2, axis=1) / C ** 2 + (nv_recv / C) ** 2) * LTT0
            nx_send = np.sum(ARM0 * self.rdata[send_SC], axis=1)
            nx_recv = np.sum(ARM0 * self.rdata[receive_SC], axis=1)
            K2 = np.sum((self.rdata[send_SC]) ** 2, axis=1) - nx_send ** 2
            P = (self.rdata[send_SC] - ARM0 * nx_send[:, np.newaxis]) / K2[:, np.newaxis]
            r_recv = np.sqrt(np.sum((self.rdata[receive_SC]) ** 2, axis=1))
            r_send = np.sqrt(np.sum((self.rdata[send_SC]) ** 2, axis=1))
            chi = P * (r_recv - r_send)[:, np.newaxis] + ARM0 * (np.log((nx_recv + r_recv) / (nx_send + r_send)))[:, np.newaxis]
            LTT2 += np.sum((2. * G * MSUN / C ** 3 * chi - (G * MSUN / 2. / C / r_recv ** 3 * LTT0 ** 2)[:, np.newaxis] * self.rdata[receive_SC]) * ARM0, axis=1)

            if pn_order == 2:
                LTTdata[label] = LTT0 + LTT1 + LTT2
            elif pn_order == 1:
                LTTdata[label] = LTT0 + LTT1 
            elif pn_order == 0:
                LTTdata[label] = LTT0 
            else: 
                raise NotImplementedError('PN order not implemented.')
            
            # LTTs are calculated at the emission times, while in the simulation we use LTTs at the reception times 
            self._LTTfunctions[label] = InterpolatedUnivariateSpline(self.tdata + LTTdata[label], LTTdata[label], k=5, ext='extrapolate')
            self._Dopplerfunctions[label] = self._LTTfunctions[label].derivative()
            
            ARMdata[label] = ARM0 + ARM1
            # arm vectors are calculated at the emission times, while in the simulation we use arm vectors at the reception times
            self._ArmVectorfunctions[label] = interp1d(self.tdata + LTTdata[label], ARMdata[label], axis=0, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        self.TCBinTPSfunctions = {}
        self.TPSinTCBfunctions = {}
        self.TPSwrtTCBfunctions = {} # TPS - TCB in TCB
        TPSdata = {}
        for label in SC_labels:
            vSC = np.sqrt(np.sum((self.vdata[label]) ** 2, axis=1))
            rSC = np.sqrt(np.sum((self.rdata[label]) ** 2, axis=1))
            rel_diff = -G * MSUN / rSC / C ** 2 - vSC ** 2 / 2. / C ** 2 
            rel_diff = cumulative_trapezoid(np.insert(rel_diff, 0, 0), dx=self.dt) # proper time = tcb at the start time 
            TPSdata[label] = self.tdata + rel_diff 
            self.TCBinTPSfunctions[label] = InterpolatedUnivariateSpline(TPSdata[label], self.tdata, k=5, ext='extrapolate')
            self.TPSinTCBfunctions[label] = InterpolatedUnivariateSpline(self.tdata, TPSdata[label], k=5, ext='extrapolate')
            self.TPSwrtTCBfunctions[label] = InterpolatedUnivariateSpline(self.tdata, TPSdata[label] - self.tdata, k=5, ext='extrapolate')
            
        # calculate ppr data, recv proper time: TPSdata, recv tcb: tdata, send tcb: tdata - LTT(tdata), send proper time: TPSinTCB(send tcb)
        self._PPRfunctions = {}
        self._DPPRfunctions = {}
        for label in MOSA_labels:
            send_SC = label[1]
            receive_SC = label[0]
            recv_tps = TPSdata[receive_SC]
            recv_tcb = self.tdata
            send_tcb = recv_tcb - self._LTTfunctions[label](recv_tcb)
            send_tps = self.TPSinTCBfunctions[send_SC](send_tcb)
            ppr = recv_tps - send_tps
            self._PPRfunctions[label] = InterpolatedUnivariateSpline(recv_tps, ppr, k=5, ext='extrapolate')
            self._DPPRfunctions[label] = self._PPRfunctions[label].derivative()
            
        for label in SC_labels:
            self._Positionfunctions[label] = interp1d(self.tdata, self.rdata[label], axis=0, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self._Velocityfunctions[label] = interp1d(self.tdata, self.vdata[label], axis=0, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
    def ListMembers(self):
        for name, value in vars(self).items():
            print('%s=%s'%(name, value))
            
    def LTTfunctions(self):
        return self._LTTfunctions
    
    def Dopplerfunctions(self):
        return self._Dopplerfunctions
    
    def PPRfunctions(self):
        return self._PPRfunctions
    
    def DPPRfunctions(self):
        return self._DPPRfunctions
    
    def ArmVectorfunctions(self):
        return self._ArmVectorfunctions
    
    def Positionfunctions(self):
        return self._Positionfunctions
    
    def Velocityfunctions(self):
        return self._Velocityfunctions
    
class EqualArmAnalyticOrbit():
    """
    calculate the equal arm analytic orbit of LISA/Taiji
    
    Attributes:
        L: norminal armlength 
        kap, lam: initial conditions of the orbit

    index of array:
    length_vector: link, t
    arm_vector: link, xyz, t
    center_position: xyz, t
    position / relative_position: S/C, xyz, t
    all in SI unit
    """
    def __init__(self, L = L_nominal, a = AU, kap = 0, lam = 0):
        self.L = L
        self.LTT = self.L / C
        self.a = a
        self.kap = kap
        self.lam = lam
        self.e = L / 2. / a / np.sqrt(3.)

    def position(self, t):
        A = 2. * np.pi / 365. / DAY * t + self.kap
        position_t = []
        for n in range(0, 3):
            Bn = n * 2. * np.pi / 3. + self.lam
            xn = self.a * np.cos(A) + \
                self.a * self.e * (np.sin(A) * np.cos(A) * np.sin(Bn) \
                                    - (1. + (np.sin(A)) ** 2) * np.cos(Bn))
            yn = self.a * np.sin(A) + \
                self.a * self.e * (np.sin(A) * np.cos(A) * np.cos(Bn) \
                                    - (1. + (np.cos(A)) ** 2) * np.sin(Bn))
            zn = -np.sqrt(3.) * self.a * self.e * np.cos(A - Bn)
            position_t.append([xn, yn, zn])
        return np.array(position_t)

    def center_position(self, t):
        A = 2. * np.pi / 365. / DAY * t + self.kap
        xn = self.a * np.cos(A) 
        yn = self.a * np.sin(A) 
        zn = np.zeros_like(xn)
        return np.array([xn, yn, zn])

    def relative_position(self, t):
        A = 2. * np.pi / 365. / DAY * t + self.kap
        position_t = []
        for n in range(0, 3):
            Bn = n * 2. * np.pi / 3. + self.lam
            xn = self.a * self.e * (np.sin(A) * np.cos(A) * np.sin(Bn) \
                                    - (1. + (np.sin(A)) ** 2) * np.cos(Bn))
            yn = self.a * self.e * (np.sin(A) * np.cos(A) * np.cos(Bn) \
                                    - (1. + (np.cos(A)) ** 2) * np.sin(Bn))
            zn = -np.sqrt(3.) * self.a * self.e * np.cos(A - Bn)
            position_t.append([xn, yn, zn])
        return np.array(position_t)

    def arm_vector(self, t): # 1, 1s, 2, 2s, 3, 3s
        p1, p2, p3 = self.position(t)
        L1 = p2 - p3
        L2 = p3 - p1
        L3 = p1 - p2
        n1 = np.zeros_like(L1)
        n2 = np.zeros_like(L2)
        n3 = np.zeros_like(L3)
        for i in range(len(L1[0])):
            n1[:, i] = L1[:, i] / np.linalg.norm(L1[:, i])
            n2[:, i] = L2[:, i] / np.linalg.norm(L2[:, i])
            n3[:, i] = L3[:, i] / np.linalg.norm(L3[:, i])
        return np.array([n1, -n1, n2, -n2, n3, -n3])

    def __call__(self, times):
        Li = np.ones((6, len(times))) * self.L
        ni = self.arm_vector(times)
        R0i = self.center_position(times)
        R0SCi = self.relative_position(times)
        return times, Li, ni, R0i, R0SCi
    

