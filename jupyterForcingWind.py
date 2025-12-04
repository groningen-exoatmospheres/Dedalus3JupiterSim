"""
Dedalus script simulating the viscous shallow water equations on a sphere. This
script demonstrates solving an initial value problem on the sphere. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_sphere.py` script can be used to produce
plots from the saved data. The simulation should about 5 cpu-minutes to run.
The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP.
To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shallow_water.py
    $ mpiexec -n 4 python3 plot_sphere.py snapshots/*.h5
    
    
Changes to this script for example listed on Dedalus Website:
Sphere parameters changed to match those of Jupiter (radius, gravity, scale height, diffusion, Omega).
Timestep scaled down to 1 second to test for stability
Quadrature grid resolution increased to 2048x1024
Velocity field is constructed using JWST data, interpolated to the quadrature grid resolution (1024 datapoints)
Previous tests using data as direct input of the velocity field have resulted in 
the simulation crashing through what is expected to be Gibbs ringing, likely
due to the data being discrete datapoints and as a result not having an analytical
solution.
Instead, the np.fft.fft method is used to create an array that is then parsed 
into f() to create a smooth function to approximate the data while still being analytical
This function is then windowed through erfWindow().
Perturbation of the original Dedalus script is removed to test for stability
Order 2 diffusion is used instead of order 4 hyperdiffusion
"""
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.special import erf
from dedalus.extras import flow_tools

# Simulation units
meter = 1 / 71.492e6
hour = 1
second = hour / 3600
day = 24 * hour

restart=0
checkpoint_dt= 48*hour
checkpoint_file = 'checkpoints_s275.h5'


# Parameters
Nphi = 1024
Ntheta = 512
dealias = 3/2

R_jupiter = 7.1492e7 * meter
R = 1.2 * R_jupiter  

rot_period_days = 2.0
Omega = 2.0 * np.pi / (rot_period_days * day)
nu = 1e5 * meter**2 / second #/ 32**2 # Diffusion matched at ell=32
g = 10 * meter / (second**2)

kB = 1.380649e-23
mu_si = 2.0 * 1.66053906660e-27   # mean molecular mass [kg]
T_eq = 1500.0                     
H_scale = (kB * T_eq) / (mu_si * g)  # scale height [m]

aveH = 5.0 * H_scale

DeltaH = 0.1 * aveH
radTimescale = 12.0 * hour

timestep = 20 * second
stop_sim_time = 2000 * hour
dtype = np.float64


# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases=basis)
hEq = dist.Field(name='hEq', bases=basis)

t = dist.Field()

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))


# Initial conditions: zonal jet
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
windowLat = np.pi/7    

# Positive Jets
umax = 35 * meter / second
latM = 35
latW = 10
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u_jet3
umax = 150 * meter / second
latM = 25
latW = 15
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3
umax = 115 * meter / second
latM = 0
latW = 20
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3
umax = 50 * meter / second
latM = -35
latW = 10
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3

# Negative Jets
umax = -20 * meter / second
latM = 30
latW = 10
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3
umax = -20 * meter / second
latM = 15
latW = 10
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3
umax = -65 * meter / second
latM = -20
latW = 10
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3
umax = -25 * meter / second
latM = -30
latW = 10
lat0 = (latM + latW)*np.pi/180
lat1 = (latM - latW)*np.pi/180
en = np.exp(-4 / (lat1 - lat0)**2)
jet3 = (lat1 <= lat) * (lat <= lat0)
u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3


# Additional Jets (Details)
def gaussBand(umaxR, latM, latW):
    umax = umaxR * meter / second
    lat0 = (latM + latW)*np.pi/180
    lat1 = (latM - latW)*np.pi/180
    en = np.exp(-4 / (lat1 - lat0)**2)
    jet3 = (lat1 <= lat) * (lat <= lat0)
    u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
    u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3
    
gaussBand(38,-27,15)
gaussBand(35,-43,10)
gaussBand(35,-52,10)
gaussBand(20,-61,10)
gaussBand(38,-67,10)
gaussBand(20,43,10)
gaussBand(15,47,10)
gaussBand(20,57,10)
gaussBand(25,66,12)

# Risky Bands (High slope)

gaussBand(60,-7,10)
gaussBand(70,10,12)




# Initial conditions: balanced height
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h) = 0")
solver = problem.build_solver()
solver.solve()

 # For reproducibility
noise_amplitude = 1e2 * meter  # ~1000m amplitude
h['g'] += noise_amplitude * (np.random.rand(*h['g'].shape) - 0.5)

#noise_velocity = 10 * meter / second #
#u['g'][0] += noise_velocity * (np.random.rand(*u['g'][0].shape) - 0.5)


# Initial conditions: equilibrium height
# Use the height fraction to get the maximum deviation from the nightside height

phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
heq_val = DeltaH * np.cos(lat) * np.cos(phi)
hEq['g'] = heq_val

# Rampup curve for the height forcing to emulate planet migration
#saturationCurve = 1 #/(1+np.exp(-0.05*(t-60)))


# Problem
problem = d3.IVP([u, h], time=t, namespace=locals())
problem.add_equation("dt(u) - nu*(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
problem.add_equation("dt(h) - nu*(lap(h)) + aveH*div(u) = (hEq-h)/radTimescale - div(h*u)")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    # delta.fill_random('g', seed=42, distribution='normal', scale=1e-1) # Random noise
    # eta.fill_random('g', seed=42, distribution='normal', scale=1e-1) # Random noise
    #eta.low_pass_filter(scales=0.5)
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state('./checkpoints/'+checkpoint_file)
    file_handler_mode = 'append'


# Analysis
snapshots = solver.evaluator.add_file_handler('jupiterGauss', sim_dt=10*hour, max_writes=10)
snapshots.add_task(h, name='height')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

checkpoints = solver.evaluator.add_file_handler('checkpoints', 
                                                sim_dt=checkpoint_dt, 
                                                max_writes=1, 
                                                mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()