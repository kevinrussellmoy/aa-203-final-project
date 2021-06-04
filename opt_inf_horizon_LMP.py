# Optimize over an infinite horizon (e.g. run the entire optimization at once, rather than a fixed horizon as with MPC)
# LMP ONLY
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

# Import LMP rate data; convert to numpy array
df_lmp = pd.read_csv("df_LMP.csv")
lmp = df_lmp.LMP_kWh.to_numpy()
lmp = lmp.repeat(4)[:load.size]  # to ensure that all data arrays are the same length

# %% Select length of optimization (1 week at first)

# define vector length of horizonfor MPC
opt_len = 24 * 7 * int(1/HR_FRAC)

opt_start = 0
opt_end = opt_start + opt_len
load_opt = load[opt_start:opt_end]
tariff_opt = tariff[opt_start:opt_end]
lmp_opt = lmp[opt_start:opt_end]
times_opt = times[opt_start:opt_end]

#%% TOU + LMP Optimization configuration

# Create a new model
m = gp.Model('lmp')
m.Params.LogToConsole = 0  # suppress console output

# Create variables:

# each LMP power flow
# ESS Power dispatched to LMP (positive=discharge, negative=charge)
ess_c_lmp = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='ess_c_lmp')
ess_d_lmp = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='ess_d_lmp')

# Integer indicator variables
chg_bin_lmp = m.addMVar(opt_len, vtype=GRB.BINARY, name='chg_bin_lmp')
dch_bin_lmp = m.addMVar(opt_len, vtype=GRB.BINARY, name='dch_bin_lmp')

#Energy stored in ESS
ess_E = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='E')

# Constrain initlal and final stored energy in battery
m.addConstr(ess_E[0] == BAT_KWH_INIT)
m.addConstr(ess_E[opt_len-1] == BAT_KWH_INIT)

for t in range(opt_len):
    # ESS power constraints
    m.addConstr(ess_c_lmp[t] <= BAT_KW * chg_bin_lmp[t])
    m.addConstr(ess_d_lmp[t] <= BAT_KW * dch_bin_lmp[t])
    m.addConstr(ess_c_lmp[t] >= 0)
    m.addConstr(ess_d_lmp[t] >= 0)

    # ESS energy constraints
    m.addConstr(ess_E[t] <= BAT_KWH_MAX)
    m.addConstr(ess_E[t] >= BAT_KWH_MIN) 

    # #Ensure non-simultaneous charge and discharge across all LMP and TOU
    m.addConstr(chg_bin_lmp[t] + dch_bin_lmp[t] <= 1)

# Time evolution of stored energy
for t in range(1,opt_len):
    m.addConstr(ess_E[t] == HR_FRAC*(ess_c_lmp[t-1]) + ess_E[t-1] - HR_FRAC*(ess_d_lmp[t-1]))

# Prohibit power flow at the end of the horizon (otherwise energy balance is off)
m.addConstr(ess_d_lmp[opt_len-1] == 0)
m.addConstr(ess_c_lmp[opt_len-1] == 0)

# Objective function
m.setObjective(HR_FRAC*(sum(lmp_opt[i] * (ess_d_lmp[i] - ess_c_lmp[i]) for i in range(opt_len))), GRB.MAXIMIZE)

# Solve the optimization
# m.params.NonConvex = 2
m.params.MIPGap = 2e-3

t = time.time()
m.optimize()
elapsed = time.time() - t

print("Elapsed time for 1 week of optimization: {}".format(elapsed))
# Elapsed time for 1 week of optimization (TOU + LMP): 6.510379076004028
                            
lmp_run = HR_FRAC * (ess_d_lmp.X-ess_c_lmp.X) * lmp_opt
tou_run = HR_FRAC * load_opt * tariff_opt
# rev = sum(lmp_run)
# cost = sum(tou_run)
# print("LMP Revenue")
# print(rev)
# print("TOU Cost")
# print(cost)

print("\n")
print("Cumulative revenue:")
print(np.sum(lmp_run))
# Cumulative revenue: 10.79
# Cumulative TOU: 4.65
# Cumulative profit(revenue - TOU): 6.14

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
p1 = ax1.plot(times_plt, lmp_run - tou_run)
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
p1 = ax1.plot(times_plt, np.cumsum(np.array(lmp_run - tou_run)))
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
ax1.set_ylim([-0.15, 0.15])
ax1.set_title("LMP and LMP dispatch")
color = 'tab:red'
p1 = ax1.plot(times_opt, lmp_opt, color=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("Power, kW")
p2 = ax2.plot(times_plt, ess_d_lmp.X - ess_c_lmp.X, color=color)
plt.grid()

# %%
