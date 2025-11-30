import numpy as np 
from scipy.integrate import cumulative_simpson, cumulative_trapezoid

from Triangle.GW import GeneralTDIResponse, FastMichelsonTDIResponse
from Triangle.Constants import * 
from Triangle.Orbit import * 
from Triangle.TDI import TDIStringManipulation, AETfromXYZ
from Triangle.Glitch import * 


class Glitch:
    short_glitch_kwargs = dict(dv=2.2e-12, tau1=10.0, tau2=11.0)
    long_glitch_kwargs = dict(dv=1.18e-12, tau1=5661.65, tau2=5661.71)

    def __init__(self, fsample=1.0):
        self.fsample = fsample

    def LPF_legacy_glitch_model(self, t, dv, tau1, tau2, t0=0):
        """
        LPF glitch model in the unit of acceleration [m/s2]
        """
        res = np.zeros_like(t)
        inds = np.where(t >= t0)[0]
        t = t[inds]
        res[inds] = dv / (tau1 - tau2) * (np.exp(-(t - t0) / tau1) - np.exp(-(t - t0) / tau2)) * np.heaviside(t - t0, 1)
        # res = dv / (tau1 - tau2) * (np.exp(-(t - t0) / tau1) - np.exp(-(t - t0) / tau2)) 
        
        return res

    def acc2ffd(self, data):
        """
        convert acceleration to fractional frequency difference, which can be then used as Interferometer.BasicNoise["acc_noise"]
        """
        return cumulative_trapezoid(np.insert(data, 0, 0), dx=1 / self.fsample) / C
    

class GlitchInGeneralTDI(GeneralTDIResponse): 
    def __init__(self, orbit, TDIstring, tcb_times):
        
        Pstring=TDIstring
        use_gpu=False 
        drop_points=0,
        linear_interp=False 
        return_eta=False
        
        super().__init__(orbit, Pstring, tcb_times, use_gpu, drop_points, linear_interp, return_eta)
        
        self.dt = tcb_times[1] - tcb_times[0]
        self.dij = dict()
        for key in MOSA_labels: 
            self.dij[key] = self.orbit_object.LTTfunctions()[key](self.tcb_times)
        
        
    def generate_test_mass_glitch(self, glitch_generator, parameters, location="12", start_time_index=None, end_time_index=None, drop_points=None): 
        """ 
        Parameters: 
            glitch_generator: a function that takes time and parameter dictionary as inputs, and returns a time series of test-mass acceleration glitch
            location: the index of test mass where glitch occurs, should be one of ["12", "23", "31", "21", "32", "13"]
            since glitch is transient, one can always use star_time_index and end_time_index to specify the time duration of it (within self.tcb_times)
        Return: 
            a time series of glitch in the fractional frequency difference unit, can be inserted to the full time series of some TDI observable as: 
            TDI[start_time_index : end_time_index+1] += output of this function
        """
        if location not in MOSA_labels: 
            raise ValueError('Invalid glitch location. The value should be one of ["12", "23", "31", "21", "32", "13"].')
        if start_time_index is None: 
            start_time_index = 0
        if end_time_index is None: 
            end_time_index = len(self.tcb_times) - 1 
            
        # glitch at TM_ij affects eta_ij and eta_ji:
        # eta_ij += delta_ij 
        # eta_ji += D_ji delta_ij 
        ij = location
        ji = ij[1]+ij[0]
        print("ij =", ij, "ji=", ji)
        
        glitch_time_in_TDI = self.tcb_times[start_time_index:end_time_index+1]
        acc_glitch_in_TDI = np.zeros_like(glitch_time_in_TDI)
        
        if self.Ndelay_dict[ij] > 0:
            print("has contribution from", ij)
            for Idelay in range(self.Ndelay_dict[ij]): 
                delayed_time = glitch_time_in_TDI - self.delay_dict[ij][Idelay][start_time_index:end_time_index+1]
                acc_glitch_in_TDI += self.delay_factor_dict[ij][Idelay] * glitch_generator(delayed_time, parameters)
                
        if self.Ndelay_dict[ji] > 0: 
            print("has contribution from", ji)
            for Idelay in range(self.Ndelay_dict[ji]): 
                delayed_time = glitch_time_in_TDI - self.delay_dict[ji][Idelay][start_time_index:end_time_index+1] - self.dij[ji][start_time_index:end_time_index+1]
                acc_glitch_in_TDI += self.delay_factor_dict[ji][Idelay] * glitch_generator(delayed_time, parameters)
        
        # assume time delay and unit conversion are commutative since the delays vary slowly compared to the transient time scale of glitches 
        # ffd_glitch_in_TDI = cumulative_trapezoid(y=np.insert(acc_glitch_in_TDI, 0, 0), dx=self.dt) / C 
        ffd_glitch_in_TDI = cumulative_simpson(y=np.insert(acc_glitch_in_TDI, 0, 0), dx=self.dt) / C 
        
        if drop_points is not None:
            ffd_glitch_in_TDI[:drop_points] = 0. 
            ffd_glitch_in_TDI[-drop_points:] = 0. 
        
        return ffd_glitch_in_TDI 

    
