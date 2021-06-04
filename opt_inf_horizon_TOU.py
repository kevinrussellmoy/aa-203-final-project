# Optimize over an infinite horizon (e.g. run the entire optimization at once, rather than a fixed horizon as with MPC)
# TOU ONLY
# Kevin Moy, 6/3/2021

#%%
import cvxpy as cp
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import time

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

# %% Select length of optimization (1 week at first)

# define vector length of horizonfor MPC
opt_len = 24 * 7 * int(1/HR_FRAC)

opt_start = 0
opt_end = opt_start + opt_len
load_opt = load[opt_start:opt_end]
tariff_opt = tariff[opt_start:opt_end]
times_opt = times[opt_start:opt_end]

#%% TOU Optimization configuration

# Create a new model
m = gp.Model('tou')
m.Params.LogToConsole = 0  # suppress console output

# Create variables:

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
chg_bin_tou = m.addMVar(opt_len, vtype=GRB.BINARY, name='chg_bin_tou')
dch_bin_tou = m.addMVar(opt_len, vtype=GRB.BINARY, name='dch_bin_tou')

#Energy stored in ESS
ess_E = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='E')

# Constrain initlal and final stored energy in battery
m.addConstr(ess_E[0] == BAT_KWH_INIT)
m.addConstr(ess_E[opt_len-1] == BAT_KWH_INIT)

for t in range(opt_len):
    # ESS power constraints
    m.addConstr(ess_c_tou[t] <= BAT_KW * chg_bin_tou[t])
    m.addConstr(ess_d_tou[t] <= BAT_KW * dch_bin_tou[t])
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
    m.addConstr(load_opt[t] == ess_load[t] + grid_load[t])

    # #Ensure non-simultaneous charge and discharge across all LMP and TOU
    m.addConstr(chg_bin_tou[t] + dch_bin_tou[t] <= 1)

# Time evolution of stored energy
for t in range(1,opt_len):
    m.addConstr(ess_E[t] == HR_FRAC*(ess_c_tou[t-1]) + ess_E[t-1] - HR_FRAC*(ess_d_tou[t-1]))

# Prohibit power flow at the end of the horizon (otherwise energy balance is off)
m.addConstr(ess_d_tou[opt_len-1] == 0)
m.addConstr(ess_c_tou[opt_len-1] == 0)

# Objective function
m.setObjective(-HR_FRAC*tariff_opt @ grid, GRB.MAXIMIZE)

# Solve the optimization
# m.params.NonConvex = 2
m.params.MIPGap = 2e-3

t = time.time()
m.optimize()
elapsed = time.time() - t

print("Elapsed time for 1 week of optimization: {}".format(elapsed))
# Elapsed time for 1 week of optimization (TOU + LMP): 6.510379076004028
                            
tou_run = HR_FRAC * grid.X * tariff_opt
load_run = HR_FRAC * load_opt * tariff_opt
# rev = sum(lmp_run)
# cost = sum(tou_run)
# print("LMP Revenue")
# print(rev)
# print("TOU Cost")
# print(cost)

print("\n")
print("Cumulative saving:")
print(np.sum(load_run - tou_run))
# Cumulative saving: 0.8897252068750001
# Cumulative TOU: 3.76016406
# Cumulative load: 4.649889266875

# %% Net profit from ESS
times_plt = times[:opt_len]
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
ax1.set_title("ESS Net Savings, Complete Optimization")
p1 = ax1.plot(times_plt, load_run - tou_run)
plt.grid()

# %% Cumulative profit from ESS
times_plt = times[:opt_len]
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue, $")
# ax1.set_ylim([-1,8])
# ax1.set_title("ESS Revenue, Disaggregated")
# p1 = ax1.plot(times_plt, lmp_ls)
# p2 = ax1.plot(times_plt, -tou_ls)
ax1.set_title("Cumulative ESS Savings, Complete Optimization")
p1 = ax1.plot(times_plt, np.cumsum(np.array(load_run - tou_run)))
plt.grid()

# %% Test plots!
# Net dispatch of ESS
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("LMP, $/kWh")
ax1.set_ylim([0, 0.15])
ax1.set_title("TOU and TOU dispatch")
color = 'tab:red'
p1 = ax1.plot(times_opt, tariff_opt, color=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("Power, kW")
p2 = ax2.plot(times_plt, ess_d_tou.X - ess_c_tou.X, color=color)
plt.grid()

# Load power flow disaggregation
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax1.set_title("Grid and ESS Contribution to Load")
p1 = ax1.plot(times_opt, load_opt, linewidth=4, linestyle=":")
p2 = ax1.plot(times_opt, grid.X)
p3 = ax1.plot(times_opt, ess_d_tou.X - ess_c_tou.X)
plt.legend(["Total Load Demand", "Grid Supply", "ESS Supply"])
plt.grid()

# %%
