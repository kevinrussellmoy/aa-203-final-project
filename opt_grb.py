# File to compute optimal TOU dispatch from load data and tariff rate pricing
# Kevin Moy, 6/2/2021

#%%
import cvxpy as cp
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# Set environment variables:
# LOAD_LEN = load.size  # length of optimization
BAT_KW = 5.0  # Rated power of battery, in kW, continuous power for the Powerwall
BAT_KWH = 14.0  # Rated energy of battery, in kWh.
# Note Tesla Powerwall rates their energy at 13.5kWh, but at 100% DoD,
# but I have also seen that it's actually 14kwh, 13.5kWh usable
BAT_KWH_MIN = 0.0 * BAT_KWH  # Minimum SOE of battery, 10% of rated
BAT_KWH_MAX = 1.0 * BAT_KWH  # Maximum SOE of battery, 90% of rated
BAT_KWH_INIT = 0.5 * BAT_KWH  # Starting SOE of battery, 50% of rated
HR_FRAC = (
    15 / 60
)  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh
NUM_HOURS = 24 # Number of hours in one day

# Import load and tariff rate data; convert to numpy array
df_load = pd.read_csv("load_tariff.csv")

load = df_load.gridnopv.to_numpy()
tariff = df_load.tariff.to_numpy()
times = pd.to_datetime(df_load.local_15min)

# Import LMP rate data; convert to numpy array
df_lmp = pd.read_csv("df_LMP.csv")
lmp = df_lmp.LMP_kWh.to_numpy()
lmp = lmp.repeat(4)[:load.size]  # to ensure that all data arrays are the same length

# %% Select length of optimization ( 1 day at first )
# TODO: Try horizons of:
# 2 hours ~9.2 iterations/sec, cumulative profit = 3.991 (4.3718 w/o terminal energy constraint)
# 6 hours ~3.1 iterations/sec, cumulative profit = 7.127 (7.284 w/0 term energy const)
# 1 day
num_hours_mpc = 6

# define vector length of horizon for MPC
opt_len = num_hours_mpc * int(1/HR_FRAC)

# opt_start = 1
# opt_end = opt_start + opt_len
# load_opt = load[opt_start:opt_end]
# tariff_opt = tariff[opt_start:opt_end]
# lmp_opt = lmp[opt_start:opt_end]
# times_opt = times[opt_start:opt_end]

#%% Optimization configuration

def tou_lmp_mpc(load, tariff, lmp, ess_E_0):
    # Create a new model
    m = gp.Model('tou-lmp')
    m.Params.LogToConsole = 0  # suppress console output

    # Create variables:

    # each LMP power flow
    # ESS Power dispatched to LMP (positive=discharge, negative=charge)
    ess_c_lmp = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='ess_c_lmp')
    ess_d_lmp = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='ess_d_lmp')

    # each TOU power flow
    # format: to_from
    ess_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_load')
    grid_ess = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid_ess')
    grid_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid_load')
    grid = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid')
    load_curtail = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='load_curtail')

    # ESS Power dispatch to TOU (positive=discharge, negative=charge)
    ess_c_tou = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_c_tou')
    ess_d_tou = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_d_tou')

    # Integer indicator variables
    chg_bin_lmp = m.addMVar(opt_len, vtype=GRB.BINARY, name='chg_bin_lmp')
    dch_bin_lmp = m.addMVar(opt_len, vtype=GRB.BINARY, name='dch_bin_lmp')
    chg_bin_tou = m.addMVar(opt_len, vtype=GRB.BINARY, name='chg_bin_tou')
    dch_bin_tou = m.addMVar(opt_len, vtype=GRB.BINARY, name='dch_bin_tou')

    #Energy stored in ESS
    ess_E = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='E')

    # Constrain initlal and final stored energy in battery
    # TODO: Modify this to account for MPC energy as an input
    m.addConstr(ess_E[0] == ess_E_0)
    # m.addConstr(ess_E[opt_len-1] == ess_E_0)  # can this be 0?? does this need to be constrained??

    for t in range(opt_len):
        # ESS power constraints
        m.addConstr(ess_c_lmp[t] <= BAT_KW * chg_bin_lmp[t])
        m.addConstr(ess_d_lmp[t] <= BAT_KW * dch_bin_lmp[t])
        m.addConstr(ess_c_tou[t] <= BAT_KW * chg_bin_tou[t])
        m.addConstr(ess_d_tou[t] <= BAT_KW * dch_bin_tou[t])
        m.addConstr(ess_c_lmp[t] >= 0)
        m.addConstr(ess_d_lmp[t] >= 0)
        m.addConstr(ess_c_tou[t] >= 0)
        m.addConstr(ess_d_tou[t] >= 0)

        # ESS energy constraints
        m.addConstr(ess_E[t] <= BAT_KWH_MAX)
        m.addConstr(ess_E[t] >= BAT_KWH_MIN) 

        # TOU power flow constraints
        m.addConstr(ess_c_tou[t] == grid_ess[t])
        m.addConstr(grid[t] == grid_ess[t] + grid_load[t])
        m.addConstr(ess_d_tou[t] == ess_load[t])
        # TODO: Figure out how to remove and add this constraint as load_opt changes in each iteration
        m.addConstr(load[t] == ess_load[t] + grid_load[t])

        # #Ensure non-simultaneous charge and discharge across all LMP and TOU
        m.addConstr(chg_bin_tou[t] + dch_bin_tou[t] + chg_bin_lmp[t] + dch_bin_lmp[t] <= 1)

    # Time evolution of stored energy
    for t in range(1,opt_len):
        m.addConstr(ess_E[t] == HR_FRAC*(ess_c_lmp[t-1] + ess_c_tou[t-1]) + ess_E[t-1] - HR_FRAC*(ess_d_lmp[t-1] + ess_d_tou[t-1]))

    # Prohibit power flow at the end of the horizon (otherwise energy balance is off)
    m.addConstr(ess_d_lmp[opt_len-1] == 0)
    m.addConstr(ess_c_lmp[opt_len-1] == 0)
    m.addConstr(ess_d_tou[opt_len-1] == 0)
    m.addConstr(ess_c_tou[opt_len-1] == 0)

    # Objective function
    m.setObjective(HR_FRAC*(sum(lmp[i] * (ess_d_lmp[i] - ess_c_lmp[i]) for i in range(opt_len))) - HR_FRAC*tariff @ grid, GRB.MAXIMIZE)

    # Solve the optimization
    # m.params.NonConvex = 2
    m.params.MIPGap = 2e-3
    m.optimize()
                                
    lmp_run = HR_FRAC * (ess_d_lmp.X-ess_c_lmp.X) * lmp
    tou_run = HR_FRAC * grid.X * tariff
    # rev = sum(lmp_run)
    # cost = sum(tou_run)
    # print("LMP Revenue")
    # print(rev)
    # print("TOU Cost")
    # print(cost)

    return ess_E.X[1], lmp_run[0], tou_run[0]

#%% TODO: run MPC
# Input: load, tariff, LMP, energy stored
# output: first action, resulting objective function (net revenue), new energy stored

# TODO: Run for 7 days at:
# horizon = 6 hours
# horizon = 24 hours
# horizon = 2 hours
# horizon = 15 minutes
num_days = 7
num_steps = num_days * NUM_HOURS * int(1/HR_FRAC)

# Optional: second week of optimization
week2_start = 24*4*7*5

# Store ESS, LMP revenue, TOU cost for eacn time step
ess_E_ls = np.zeros(1)
lmp_ls = np.zeros(1)
tou_ls = np.zeros(1)

# Track energy stored in ESS
ess_E = BAT_KWH_INIT
ess_E_ls[0] = ess_E
for i in tqdm(range(num_steps)):
    opt_start = i
    opt_end = opt_start + opt_len
    load_opt = load[opt_start:opt_end]
    tariff_opt = tariff[opt_start:opt_end]
    lmp_opt = lmp[opt_start:opt_end]
    times_opt = times[opt_start:opt_end]

    # Execute 1 step of MPC
    ess_E_1, lmp_0, tou_0 = tou_lmp_mpc(load_opt, tariff_opt, lmp_opt, ess_E)

    # Store MPC step results
    ess_E_ls = np.append(ess_E_ls, ess_E_1)
    lmp_ls = np.append(lmp_ls, lmp_0)
    tou_ls = np.append(tou_ls, tou_0)

    # Set energy stored for next step
    ess_E = ess_E_1

print("Cumulative profit:")
print(np.sum(lmp_ls-tou_ls))

# %% Net profit from ESS

times_plt = times[:num_steps+1]
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue, $")
# ax1.set_title("ESS Revenue, Disaggregated")
# p1 = ax1.plot(times_plt, lmp_ls)
# p2 = ax1.plot(times_plt, -tou_ls)
ax1.set_title("ESS Net Profit, {} hour horizon".format(num_hours_mpc))
p1 = ax1.plot(times_plt, lmp_ls-tou_ls)
plt.grid()

# %% Cumulative profit from ESS
times_plt = times[:num_steps+1]
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue, $")
ax1.set_ylim(-1,8)
# ax1.set_title("ESS Revenue, Disaggregated")
# p1 = ax1.plot(times_plt, lmp_ls)
# p2 = ax1.plot(times_plt, -tou_ls)
ax1.set_title("Cumulative ESS Profit, {} hour horizon".format(num_hours_mpc))
p1 = ax1.plot(times_plt, np.cumsum(lmp_ls-tou_ls))
plt.grid()



# %%
