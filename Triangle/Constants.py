# constants
PI = 3.141592653589793
TWOPI = 6.283185307179586
C = 299792458.
G = 6.67e-11
MPC = 3.0857e22
MSUN = 1.989e30
AU = 1.496e11
DAY = 24. * 3600.
YEAR = 31558149.763545603

# cosmology
OmegaM = 0.3111
H0 = 67.66

# central frequency of laser 
F_LASER = 2.816e14

# top-level parameters for the mission 
L_nominal = 3e9 # [m], nominal arm-length 
SACC_nominal = 3e-15 # [m/s2], nominal amplitude of the test-mass acceleration noise 
SOMS_nominal = 8e-12 # [m], nominal amplitude of the optical metrology noise 

# [RO] readout noise amplitudes of different interferometers 
# SRO_SCI_C_nominal = 6.35e-12 # [m], sci_c readout noise
# SRO_SCI_SB_nominal = 12.5e-12 # [m], sci_sb readout noise
# SRO_TM_C_nominal = 1.42e-12 # [m], tm_c readout noise
# SRO_REF_C_nominal = 3.32e-12 # [m], ref_c readout noise
# SRO_REF_SB_nominal = 7.9e-12 # [m], ref_sb readout noise

# [RO] readout noise grounded in 1 term, i.e. the readout noise of science interferometer, and others are all set to 0 
SRO_SCI_C_nominal = SOMS_nominal # [m], sci_c readout noise
SRO_SCI_SB_nominal = 12.5e-12 # [m], sci_sb readout noise
SRO_TM_C_nominal = 0 # [m], tm_c readout noise
SRO_REF_C_nominal = 0 # [m], ref_c readout noise
SRO_REF_SB_nominal = 0 # [m], ref_sb readout noise

# [OP] optical path noises for different paths 
# SOP_TM_LOCAL_nominal = 4.24e-12 # [m]
# SOP_REF_LOCAL_nominal = 2e-12 # [m]
# SOP_OTHER_nominal = 1e-15 # [m]

# [OP] optical path noises = 0, i.e. only create the interface for optical path noises
SOP_TM_LOCAL_nominal = 0 # [m]
SOP_REF_LOCAL_nominal = 0 # [m]
SOP_OTHER_nominal = 0 # [m]

# other noises: laser, optical bench displacement, clock jitter, fibre back link, pseudo ranging 
SLASER_nominal = 30 # [Hz], laser frequency noise 
SOB_nominal = 1e-2 # [rad], optical bench displacement noise, equivallent to a nanometer-level displacement 
SCLOCK_nominal = 6.32e-14 # [s/s], clock noise 
SBL_nominal = 3e-12 # [m], fibre back link noise 
SR_nominal = 0.9 # [m], pseudo raninging noise 

# sideband modulation noises 
SM_LEFT_nominal = 5.2e-14 # [s/s]
SM_RIGHT_nominal = 5.2e-13 # [s/s]

# index of MOSAs and SCs
MOSA_labels = ['12', '13', '23', '21', '31', '32']
left_MOSA_labels = {'12', '23', '31'}
right_MOSA_labels = {'13', '21', '32'}
adjacent_MOSA_labels = {
    '12': '13',
    '13': '12',
    '23': '21',
    '21': '23',
    '31': '32',
    '32': '31'
}
SC_labels = ['1', '2', '3']

# sideband modulation frequencies
modulation_freqs = {'12' : 2.4e9, '13' : 2.401e9, '23' : 2.4e9, '21' : 2.401e9, '31' : 2.4e9, '32' : 2.401e9}

# locking scheme
# 1. N1-LA12
lock_topology = {'12': 'primary', '13': 'adjacent', '23': 'adjacent', '21': 'distant', '31': 'distant', '32': 'adjacent'}
lock_order = ['12', '13', '31', '32', '21', '23']
fplan = {'12': -5510000.0, '13': -17130000.0, '23': 16440000.000000002, '21': -11080000.0, '31': 14740000.0, '32': -9020000.0}

# polynomial coefficients of clock drifts  
default_clock_t0 = {'1': 0, '2': 0, '3': 0} 
default_clock_y0 = {'1': 5e-8, '2': 6.25e-7, '3': -3.75e-7} 
default_clock_y1 = {'1': 1.6e-15, '2': 2e-14, '3': -1.2e-14} 
default_clock_y2 = {'1': 9e-24, '2': 6.75e-23, '3': -1.125e-22} 

# laser frequency offset
default_laser_O0 = {'12': 0., '13': 0., '23': 0., '21': 0., '31': 0., '32': 0.}

# colors for plot
MOSA_colors = {
    '12': '#E04F38', 
    '13': '#7D8F4D', 
    '23': '#3877E0',
    '21': '#662D24',
    '31': '#B3E038',
    '32': '#293E61',
}
SC_colors = {
    '1': '#E04F38',
    '2': '#3877E0',
    '3': '#B3E038'
}
ORANGE = '#ED8D5A'
BLUE = '#68BED9'
GREEN1 = '#257D8B'
GREEN2 = '#BFDFB2'
YELLOW = '#EFCE87'
RED = '#B54764'