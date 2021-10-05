# This script find an optimal mixture of smFISH experiment and flow cytometry measurement
import sys
sys.path.append("..")
import numpy as np
from utils.fim_utils import computeCriteria, logTransform

BUDGET_MAX = 10000
SMFISH_COST = 1
FLOWCYT_COST = 0.5
T_MEASUREMENT = 45
CRITERIA = "D"
#%%
with np.load(f'results/fim_flowcyt.npz') as _:
    fim_flowcyts = _['fim']
fim_flowcyt = np.mean(fim_flowcyts, axis=0)
logTransform(fim_flowcyt)
#%%
with np.load(f'results/fim_exact.npz') as _:
    fim_fish = _['fim']
logTransform(fim_fish)
#%%
m_flowcyt = fim_flowcyt[T_MEASUREMENT]
m_fish = fim_fish[T_MEASUREMENT]
obj_values = np.zeros((BUDGET_MAX//SMFISH_COST, BUDGET_MAX//FLOWCYT_COST))
for nfish in (BUDGET_MAX//SMFISH_COST):
    for nflowcyt in range((BUDGET_MAX-nfish*SMFISH_COST)//FLOWCYT_COST):
        obj_values[nfish, nflowcyt] = computeCriteria(nfish*m_fish + nflowcyt*m_flowcyt, criteria=CRITERIA)
#%%


