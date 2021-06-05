# Code to create all combined plots
# Plots for combined cumulative profit, stored energy
# Kevin Moy, 6/4/2021

#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HR_FRAC = (
    15 / 60
)  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh
NUM_HOURS = 24 # Number of hours in one day

# length of optimization
num_days = 7
num_steps = num_days * NUM_HOURS * int(1/HR_FRAC)

# Optional: second week of optimization
week1_start = 0 
week1_end = week1_start + num_steps
week2_start = 24*4*7*5
week2_end = week2_start + num_steps

# Import load and tariff rate data; convert to numpy array
df_load = pd.read_csv("load_tariff.csv")

load = df_load.gridnopv.to_numpy()
tariff = df_load.tariff.to_numpy()
times = pd.to_datetime(df_load.local_15min)
times_wk1 = times[week1_start:week1_end-1]
times_wk2 = times[week2_start:week2_end-1]

# Number of hours for horizon
num_hours_mpc = [1, 2, 6, 12]

# %% Manually create plotting arrays

fulloptwk1 = pd.read_csv("cuml_profit_fullopt.csv", index_col=None, header=0)
fulloptwk2 = pd.read_csv("cuml_profit_fullopt_wk2.csv", index_col=None, header=0)

fulloptwk1_se = pd.read_csv("stor_energy_fullopt.csv", index_col=None, header=0)
fulloptwk2_se = pd.read_csv("stor_energy_fullopt_wk2.csv", index_col=None, header=0)

li = []

for h in num_hours_mpc:
    df = pd.read_csv("cuml_profit_{}_hr.csv".format(h), index_col=None, header=0)
    li.append(df)

cuml_profit_wk1 = pd.concat(li, axis=1, ignore_index=True)

li = []
for h in num_hours_mpc:
    df = pd.read_csv("cuml_profit_{}_hr_wk2.csv".format(h), index_col=None, header=0)
    li.append(df)

cuml_profit_wk2 = pd.concat(li, axis=1, ignore_index=True)

li = []

for h in num_hours_mpc:
    df = pd.read_csv("stor_energy_{}_hr.csv".format(h), index_col=None, header=0)
    li.append(df)

stor_energy_wk1 = pd.concat(li, axis=1, ignore_index=True)

li = []
for h in num_hours_mpc:
    df = pd.read_csv("stor_energy_{}_hr_wk2.csv".format(h), index_col=None, header=0)
    li.append(df)

stor_energy_wk2 = pd.concat(li, axis=1, ignore_index=True)

# %% Cumulative profit, week 1
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue, $")
ax1.set_title("ESS Net Profit by Horizon, Week 1")
p1 = ax1.plot(times_wk1, cuml_profit_wk1[0][:-1])
p2 = ax1.plot(times_wk1, cuml_profit_wk1[1][:-1])
p3 = ax1.plot(times_wk1, cuml_profit_wk1[2][:-1])
p4 = ax1.plot(times_wk1, cuml_profit_wk1[3][:-1])
p5 = ax1.plot(times_wk1, fulloptwk1, linewidth=4, linestyle=":")
plt.legend(["1 hr", "2 hr", "6 hr", "12 hr", "Full Optimization"])
plt.grid()
# %% Cumulative profit, week 2
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue, $")
ax1.set_title("ESS Net Profit by Horizon, Week 2")
p1 = ax1.plot(times_wk2, cuml_profit_wk2[0][:-1])
p2 = ax1.plot(times_wk2, cuml_profit_wk2[1][:-1])
p3 = ax1.plot(times_wk2, cuml_profit_wk2[2][:-1])
p4 = ax1.plot(times_wk2, cuml_profit_wk2[3][:-1])
p5 = ax1.plot(times_wk2, fulloptwk2, linewidth=4, linestyle=":")
plt.legend(["1 hr", "2 hr", "6 hr", "12 hr", "Full Optimization"])
plt.grid()

# %% stored energy, week 1
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Stored Energy, kWh")
ax1.set_title("ESS Stored Energy by Horizon, Week 1")
p1 = ax1.plot(times_wk1, stor_energy_wk1[0][:-1])
p2 = ax1.plot(times_wk1, stor_energy_wk1[1][:-1])
p3 = ax1.plot(times_wk1, stor_energy_wk1[2][:-1])
p4 = ax1.plot(times_wk1, stor_energy_wk1[3][:-1])
p5 = ax1.plot(times_wk1, fulloptwk1_se, linewidth=4, linestyle=":")
plt.legend(["1 hr", "2 hr", "6 hr", "12 hr", "Full Optimization"])
plt.grid()
# %% Cumulative profit, week 2
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Stored Energy, kWh")
ax1.set_title("ESS Stored Energy by Horizon, Week 2")
p1 = ax1.plot(times_wk2, stor_energy_wk2[0][:-1])
p2 = ax1.plot(times_wk2, stor_energy_wk2[1][:-1])
p3 = ax1.plot(times_wk2, stor_energy_wk2[2][:-1])
p4 = ax1.plot(times_wk2, stor_energy_wk2[3][:-1])
p5 = ax1.plot(times_wk2, fulloptwk2_se, linewidth=4, linestyle=":")
plt.legend(["1 hr", "2 hr", "6 hr", "12 hr", "Full Optimization"])
plt.grid()


# %%
