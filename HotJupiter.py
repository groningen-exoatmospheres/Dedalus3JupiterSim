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
day =  24.0 * hour

restart=0
checkpoint_dt= 48*hour
checkpoint_file = 'checkpoints_s3.h5'

# Parameters
Nphi = 1024
Ntheta = 512
dealias = 3/2

##R = 71.492e6 * meter
R_jupiter = 7.1492e7 * meter
R = 1.2 * R_jupiter           # planetary radius [m]

##Omega = 1.7585181e-4 / second
rot_period_days = 2.0
Omega = 2.0 * np.pi / (rot_period_days * day)  # planetary rotation rate [rad/s]

nu = 1e5 * meter**2 / second # viscosity [m2/s]

g = 10.0 * meter / (second**2) # gravity [m/s2]

##H = 2.7e4 * meter
# Mean temperature and scale height estimate:
# H = k_B * T / (m * g). For H2-dominated atmosphere (mu ~ 2 amu)
kB = 1.380649e-23
mu_si = 2.0 * 1.66053906660e-27   # mean molecular mass [kg]
T_eq = 1500.0                     # equilibrium temperature [K] (tunable)
H_scale = (kB * T_eq) / (mu_si * g)  # scale height [m]

# Background fluid-layer mean depth (Hbar): choose a few scale heights
Hbar = 5.0 * H_scale

# Forcing (day-night) amplitude (equivalent height anomaly, meters)
DeltaH = 0.1 * Hbar   # amplitude of day-night equivalent height difference (tunable)

tau_force = 24.0 * hour     # relaxation timescale
timestep = 100 * second
stop_sim_time = 7200.0 * hour
dtype = np.float64

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases=basis)
heq = dist.Field(name='heq', bases=basis)

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

phi, theta = dist.local_grids(basis)
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
windowLat = np.pi/7    

# Additional Jets (Details)
def gaussBand(umaxR, latM, latW):
    umax = umaxR * meter / second
    lat0 = (latM + latW)*np.pi/180
    lat1 = (latM - latW)*np.pi/180
    en = np.exp(-4 / (lat1 - lat0)**2)
    jet3 = (lat1 <= lat) * (lat <= lat0)
    u_jet3 = umax / en * np.exp(1 / (lat[jet3] - lat0) / (lat[jet3] - lat1))
    u['g'][0][jet3]  = u['g'][0][jet3] + u_jet3

#gaussBand(35, 35, 10)
#gaussBand(150, 25, 15)
#gaussBand(115, 0, 20)
#gaussBand(50, -35, 10)
#gaussBand(-20, 30, 10)
#gaussBand(-20, 15, 10)
#gaussBand(-65, -20, 10)
#gaussBand(-25, -30, 10)
#  
#gaussBand(38,-27,15)
#gaussBand(35,-43,10)
#gaussBand(35,-52,10)
#gaussBand(20,-61,10)
#gaussBand(38,-67,10)
#gaussBand(20,43,10)
#gaussBand(15,47,10)
#gaussBand(20,57,10)
#gaussBand(25,66,12)

# Risky Bands (High slope)

#gaussBand(60,-7,10)
#gaussBand(70,10,12)

from mpi4py import MPI
from scipy.interpolate import CubicSpline

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Step 1: Prepare the array only on rank 0
if rank == 0:
    data_jupiter = np.loadtxt("wind_jupiter.txt") 
    sh = data_jupiter.shape
else:
    data_jupiter = None
    sh = None

# Step 2: Broadcast the shape of the array first
sh = comm.bcast(sh, root=0)

# Step 3: Allocate the array on all processes
if rank != 0:
    data_jupiter = np.empty(sh, dtype='float64')

# Step 4: Broadcast the full array
comm.Bcast(data_jupiter, root=0)

# Now all processes have the full array
#print(f"Rank {rank}: data_jupiter shape = {data_jupiter.shape}")

a = data_jupiter[:,0]
b = data_jupiter[:,1]*np.pi/180
spl = CubicSpline(b, a)

u['g']

#print('Interpolation done!')
#print(lat)
#print('spl(lat)', spl(lat).shape)
#print('ug', u['g'][0].shape)

u['g'][0] = u['g'][0] + spl(lat) * meter / second

# Initial conditions: balanced height
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h) = 0")
solver = problem.build_solver()
solver.solve()

# Initial conditions: perturbation
lat2 = 0
hpert = 12000 * meter
alpha = 1 / 3
beta = 1 / 15
#h['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)

noise_amplitude = 1e2 * meter  # ~1000m amplitude
h['g'] += noise_amplitude * (np.random.rand(*h['g'].shape) - 0.5)

#noise_velocity = 10 * meter / second #
#u['g'][0] += noise_velocity * (np.random.rand(*u['g'][0].shape) - 0.5)


phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
heq_val = DeltaH * np.cos(lat) * np.cos(phi)
heq['g'] = heq_val

# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) - nu*(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
problem.add_equation("dt(h) - nu*(lap(h)) + Hbar*div(u) = - div(h*u) - (h - heq)/tau_force")

# Solver
solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state('checkpoints/'+checkpoint_file)
    file_handler_mode = 'append'

# Analysis
snapshots = solver.evaluator.add_file_handler('jupiterGauss', sim_dt=5*hour, max_writes=10)
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