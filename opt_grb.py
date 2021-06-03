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

# Import load and tariff rate data; convert to numpy array
df_load = pd.read_csv("load_tariff.csv")

load = df_load.gridnopv.to_numpy()
tariff = df_load.tariff.to_numpy()
times = pd.to_datetime(df_load.local_15min)

# Import LMP rate data; convert to numpy array
df_lmp = pd.read_csv("df_LMP.csv")
lmp = df_lmp.LMP_kWh.to_numpy()
lmp = lmp.repeat(4)[:load.size]  # to ensure that all data arrays are the same length

# %% Configure optimization

# Select length of optimization ( 1 day at first )
opt_len = 24 * 4
load_opt = load[:opt_len]
tariff_opt = tariff[:opt_len]
lmp_opt = lmp[:opt_len]
times_opt = times[:opt_len]

# Create a new model
m = gp.Model('tou-lmp')

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
m.addConstr(ess_E[0] == BAT_KWH_INIT)
m.addConstr(ess_E[opt_len-1] == BAT_KWH_INIT)

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
    m.addConstr(load_opt[t] == ess_load[t] + grid_load[t])
    m.addConstr(ess_c_tou[t] == grid_ess[t])
    m.addConstr(grid[t] == grid_ess[t] + grid_load[t])
    m.addConstr(ess_d_tou[t] == ess_load[t])

    # #Ensure non-simultaneous charge and discharge across all LMP and TOU
    m.addConstr(chg_bin_tou[t] + dch_bin_tou[t] + chg_bin_lmp[t] + dch_bin_lmp[t] <= 1)
    # m.addConstr(chg_bin_lmp[t] + dch_bin_lmp[t] <= 1)


    # Time evolution of stored energy
for t in range(1,opt_len):
    m.addConstr(ess_E[t] == HR_FRAC*(ess_c_lmp[t-1] + ess_c_tou[t-1]) + ess_E[t-1] - HR_FRAC*(ess_d_lmp[t-1] + ess_d_tou[t-1]))

# m.addConstrs(0 == ess_d[i] @ ess_c[i] for i in range(week_len))
m.addConstr(ess_d_lmp[opt_len-1] == 0)
m.addConstr(ess_c_lmp[opt_len-1] == 0)
m.addConstr(ess_d_tou[opt_len-1] == 0)
m.addConstr(ess_c_tou[opt_len-1] == 0)


# Objective function
m.setObjective(HR_FRAC*(sum(lmp_opt[i] * (ess_d_lmp[i] - ess_c_lmp[i]) for i in range(opt_len))) - tariff_opt @ grid, GRB.MAXIMIZE)
# m.setObjective(HR_FRAC * (lmp_wk @ ess_p), GRB.MAXIMIZE


# %% Solve the optimization
# m.params.NonConvex = 2
m.params.MIPGap = 2e-3
m.optimize()

# %% Net dispatch of ESS

# plt.plot(ess_d_lmp.X - ess_c_lmp.X + ess_d_tou.X - ess_c_tou.X)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax1.set_title("Net ESS Dispatch")
p1 = ax1.plot(times_opt, ess_d_lmp.X - ess_c_lmp.X + ess_d_tou.X - ess_c_tou.X)

# %% Disaggregate by application (LMP vs. TOU)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax1.set_title("Disaggregated ESS Dispatch")
p1 = ax1.plot(times_opt, ess_d_lmp.X - ess_c_lmp.X)
p2 = ax1.plot(times_opt, ess_d_tou.X - ess_c_tou.X)
plt.legend(["LMP", "TOU"])
# %%

rev = (ess_d_lmp.X-ess_c_lmp.X) @ lmp_opt
print(rev)
# %%
cost = grid.X @ tariff_opt
print(cost)
# %%
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

#%%
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
p3 = ax2.plot(times_opt, ess_d_lmp.X - ess_c_lmp.X, color=color)
ax2.set_ylabel("Power, kW")

# %%
