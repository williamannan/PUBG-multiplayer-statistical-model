"""
- The simulation study is quite expensive due to the multiple cross validation.
- This complecity limit ability to run refine domain partition on local computer.
- The 250 repeated simulation study with 80 lambda domain partition was run on super-computer
- The Excel result is found in the folder named "Sim_Excle"
- This function import all the csv files in this folder and generate the various plots seen 
  under the simulation study in the paper.
"""


####### ================================================================
####### ------- Importing packages needed for simulation ---------------
####### ================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv, LinAlgError
from tabulate import tabulate
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import re
import time
import os

####### ================================================================
####### -------- Local working directory -------------------------------
####### ================================================================
'''
To rund this code, remember to change this to the working directory on your local computer.
Failure to do so will render error.
'''
os.chdir('/Users/williameboannan/Documents/Programming/PUBG_GitHub_Code')


##### ==================================================================
#### --------- Importing the .csv results ------------------------------
##### ==================================================================
def get_stack_match_data():
    '''This function extract the all the excell results'''
    ## Get the folder directory
    data_folder = "/Users/williameboannan/Documents/Programming/PUBG_GitHub_Code/Sim_Excel"

    ## Getting the .csv files in this directory
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    ### A dictionary to store the imported csv files
    Sim_CSV_Results = {}

    for f in csv_files:
        full_path = os.path.join(data_folder, f)

        key_name = f.replace('.csv', '')

        Sim_CSV_Results[key_name] = pd.read_csv(full_path).dropna()

    return Sim_CSV_Results

###### ------------------------------------------------------------------
Sim_CSV_Results = get_stack_match_data( )                 ## Export all the csv files 

DataSize = Sim_CSV_Results['El_Sim_CombMod1']['Num_Stacked_Data']   ##  Data sizes


log_Calibrations = 8        ## Calibration on the y-axis

true_beta = -1.25
true_gamma = 1.50
alpha_max = [0.5, 2.0, 4.0]

##### ===================================================================
##### -- Effect of alpha_max and CV on RMSE between alphas --------------
##### ===================================================================

####*********************************************************************
### ------ RMSE: Elim. Data + Elim. Model, alpha_max = 0.5 --------------
unp_El_Sim_ElMod1_Alpha_Rmse = Sim_CSV_Results['El_Sim_ElMod1']['unpen_RMSE_between_alpha']/alpha_max[0]
CV_El_Sim_ElMod1_Alpha_Rmse = Sim_CSV_Results['El_Sim_ElMod1']['cv_RMSE_between_alpha']/alpha_max[0]

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_El_Sim_ElMod1_Alpha_Rmse, linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_El_Sim_ElMod1_Alpha_Rmse, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
ymin, ymax = plt.ylim([0.05,9.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 0.50$', title_fontsize=13)

plt.title(f"Elimination Model", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('ElSim_ElMod1_RMSE_Alpha.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ RMSE: Elim. Data + Elim. Model, alpha_max = 2.0 --------------
unp_El_Sim_ElMod2_Alpha_Rmse = Sim_CSV_Results['El_Sim_ElMod2']['unpen_RMSE_between_alpha']/alpha_max[1]
CV_El_Sim_ElMod2_Alpha_Rmse = Sim_CSV_Results['El_Sim_ElMod2']['cv_RMSE_between_alpha']/alpha_max[1]

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_El_Sim_ElMod2_Alpha_Rmse, linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_El_Sim_ElMod2_Alpha_Rmse, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
ymin, ymax = plt.ylim([0.05,9.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title(f"Elimination Model", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('ElSim_ElMod2_RMSE_Alpha.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ RMSE: Elim. Data + Elim. Model, alpha_max = 4.0 --------------
unp_El_Sim_ElMod3_Alpha_Rmse = Sim_CSV_Results['El_Sim_ElMod3']['unpen_RMSE_between_alpha']/alpha_max[2]
CV_El_Sim_ElMod3_Alpha_Rmse = Sim_CSV_Results['El_Sim_ElMod3']['cv_RMSE_between_alpha']/alpha_max[2]

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_El_Sim_ElMod3_Alpha_Rmse, linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_El_Sim_ElMod3_Alpha_Rmse, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
ymin, ymax = plt.ylim([0.05,9.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 4.0$', title_fontsize=13)

plt.title(f"Elimination Model", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('ElSim_ElMod3_RMSE_Alpha.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)


####*********************************************************************
### ------ RMSE: Pairwise Data + Pairwise. Model, alpha_max = 0.5 -----------
unp_Pw_Sim_PwMod1_Alpha_Rmse = Sim_CSV_Results['Pw_Sim_PwMod1']['unpen_RMSE_between_alpha']/alpha_max[0]
CV_Pw_Sim_PwMod1_Alpha_Rmse = Sim_CSV_Results['Pw_Sim_PwMod1']['cv_RMSE_between_alpha']/alpha_max[0]

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_Pw_Sim_PwMod1_Alpha_Rmse, linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_Pw_Sim_PwMod1_Alpha_Rmse, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
# ymin, ymax = plt.ylim()
plt.ylim([0.05, 40])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 0.50$', title_fontsize=13)
plt.title(f"Pairwise Model", fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_PwMod1_RMSE_Alpha.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ RMSE: Pairwise Data + Pairwise. Model, alpha_max = 2.0 -------
unp_Pw_Sim_PwMod2_Alpha_Rmse = Sim_CSV_Results['Pw_Sim_PwMod2']['unpen_RMSE_between_alpha']/alpha_max[1]
CV_Pw_Sim_PwMod2_Alpha_Rmse = Sim_CSV_Results['Pw_Sim_PwMod2']['cv_RMSE_between_alpha']/alpha_max[1]

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_Pw_Sim_PwMod2_Alpha_Rmse, linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_Pw_Sim_PwMod2_Alpha_Rmse, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
# ymin, ymax = plt.ylim()
plt.ylim([0.05, 40])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)
plt.grid(True)
plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title(f"Pairwise Model", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('PwSim_PwMod2_RMSE_Alpha.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ RMSE: Pairwise Data + Pairwise. Model, alpha_max = 4.0 -------
unp_Pw_Sim_PwMod3_Alpha_Rmse = Sim_CSV_Results['Pw_Sim_PwMod3']['unpen_RMSE_between_alpha']/alpha_max[2]
CV_Pw_Sim_PwMod3_Alpha_Rmse = Sim_CSV_Results['Pw_Sim_PwMod3']['cv_RMSE_between_alpha']/alpha_max[2]

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_Pw_Sim_PwMod3_Alpha_Rmse, linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_Pw_Sim_PwMod3_Alpha_Rmse, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
# ymin, ymax = plt.ylim()
plt.ylim([0.05, 40])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 4.0$', title_fontsize=13)
plt.title(f"Pairwise Model", fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_PwMod3_RMSE_Alpha.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)


##### ===================================================================
##### -- Effect of alpha_max and CV on MAE Rankings ---------------------
##### ===================================================================

####*********************************************************************
### ------ MAE: Elim. Data + Elim. Model, MAE ranking = 0.5 -------------
unp_El_Sim_ElMod1_MAE_Ranks = Sim_CSV_Results['El_Sim_ElMod1']['unpen_pred_MAE']
CV_El_Sim_ElMod1_MAE_Ranks = Sim_CSV_Results['El_Sim_ElMod1']['cv_pred_MAE']


plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_El_Sim_ElMod1_MAE_Ranks , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_El_Sim_ElMod1_MAE_Ranks, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
ymin, ymax = plt.ylim([0.3, 5.0])
# plt.ylim([0.05, ymax])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title(f"Elimination Model", fontweight='bold', fontsize=13)
plt.grid(True)

plt.legend(title='$\\alpha_{max} = 0.50$', title_fontsize=13)

plt.tight_layout()
plt.savefig('ElSim_ElMod1_MAE_rank.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ MAE: Elim. Data + Elim. Model, MAE ranking = 2.0 -------------
unp_El_Sim_ElMod2_MAE_Ranks = Sim_CSV_Results['El_Sim_ElMod2']['unpen_pred_MAE']
CV_El_Sim_ElMod2_MAE_Ranks = Sim_CSV_Results['El_Sim_ElMod2']['cv_pred_MAE']

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_El_Sim_ElMod2_MAE_Ranks , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_El_Sim_ElMod2_MAE_Ranks, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
ymin, ymax = plt.ylim([0.3, 5.0])
# plt.ylim([0.05, ymax])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title(f"Elimination Model", fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)

plt.tight_layout()
plt.savefig('ElSim_ElMod2_MAE_rank.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ MAE: Elim. Data + Elim. Model, MAE ranking = 4.0 -------------
unp_El_Sim_ElMod3_MAE_Ranks = Sim_CSV_Results['El_Sim_ElMod3']['unpen_pred_MAE']
CV_El_Sim_ElMod3_MAE_Ranks = Sim_CSV_Results['El_Sim_ElMod3']['cv_pred_MAE']

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_El_Sim_ElMod3_MAE_Ranks , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_El_Sim_ElMod3_MAE_Ranks, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
ymin, ymax = plt.ylim([0.3, 5.0])
# plt.ylim([0.05, ymax])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title(f"Elimination Model", fontweight='bold', fontsize=13)
plt.grid(True)

plt.legend(title='$\\alpha_{max} = 4.0$', title_fontsize=13)

plt.tight_layout()
plt.savefig('ElSim_ElMod3_MAE_rank.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ MAE: Pairwise Data + Pairwise Model, MAE ranking = 0.5 -------------
unp_Pw_Sim_PwMod1_MAE_Ranks = Sim_CSV_Results['Pw_Sim_PwMod1']['unpen_pred_MAE']
CV_Pw_Sim_PwMod1_MAE_Ranks = Sim_CSV_Results['Pw_Sim_PwMod1']['cv_pred_MAE']

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_Pw_Sim_PwMod1_MAE_Ranks , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_Pw_Sim_PwMod1_MAE_Ranks, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.3, 5.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title(f"Pairwise Model", fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 0.50$', title_fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_PwMod1_MAE_rank.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ MAE: Pairwise Data + Pairwise Model, MAE ranking = 2.0 -------------
unp_Pw_Sim_PwMod2_MAE_Ranks = Sim_CSV_Results['Pw_Sim_PwMod2']['unpen_pred_MAE']
CV_Pw_Sim_PwMod2_MAE_Ranks = Sim_CSV_Results['Pw_Sim_PwMod2']['cv_pred_MAE']

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_Pw_Sim_PwMod2_MAE_Ranks , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_Pw_Sim_PwMod2_MAE_Ranks, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.3, 5.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title(f"Pairwise Model", fontweight='bold', fontsize=13)
plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_PwMod2_MAE_rank.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ MAE: Pairwise Data + Pairwise Model, MAE ranking = 4.0 -------------
unp_Pw_Sim_PwMod3_MAE_Ranks = Sim_CSV_Results['Pw_Sim_PwMod3']['unpen_pred_MAE']
CV_Pw_Sim_PwMod3_MAE_Ranks = Sim_CSV_Results['Pw_Sim_PwMod3']['cv_pred_MAE']

plt.figure(figsize=(5,3))
plt.plot(DataSize, unp_Pw_Sim_PwMod3_MAE_Ranks , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Unpen")
plt.plot(DataSize, CV_Pw_Sim_PwMod3_MAE_Ranks, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="CV")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.3, 5.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title(f"Pairwise Model", fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 4.0$', title_fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_PwMod3_MAE_rank.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

########=================================================================
######## ---- Comparing the RMSE for the three models -------------------
########=================================================================

####*********************************************************************
### ------ Elimination data ------ --------------------------------------
El_Sim_ElMod2_RMSE_Alpha = Sim_CSV_Results['El_Sim_ElMod2']['cv_RMSE_between_alpha']/alpha_max[1]
El_Sim_PwMod2_RMSE_Alpha = Sim_CSV_Results['El_Sim_PwMod2']['cv_RMSE_between_alpha']/alpha_max[1]
El_Sim_CbMod2_RMSE_Alpha = Sim_CSV_Results['El_Sim_CombMod2']['cv_RMSE_between_alpha']/alpha_max[1]

plt.figure(figsize=(6,4))
plt.plot(DataSize, El_Sim_ElMod2_RMSE_Alpha , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Elimination model")
plt.plot(DataSize, El_Sim_PwMod2_RMSE_Alpha, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="Pairwise model")
plt.plot(DataSize, El_Sim_CbMod2_RMSE_Alpha, linewidth=2.5, marker='X', color='r',linestyle='-.', 
         label="Composite model")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.05, 1.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title("Elimination Data", fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('ElSim_RMSE_AllModel.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ Pairwise Data ---------- --------------------------------------
Pw_Sim_ElMod2_RMSE_Alpha = Sim_CSV_Results['Pw_Sim_ElMod2']['cv_RMSE_between_alpha']/alpha_max[1]
Pw_Sim_PwMod2_RMSE_Alpha = Sim_CSV_Results['Pw_Sim_PwMod2']['cv_RMSE_between_alpha']/alpha_max[1]
Pw_Sim_CbMod2_RMSE_Alpha = Sim_CSV_Results['Pw_Sim_CombMod2']['cv_RMSE_between_alpha']/alpha_max[1]

plt.figure(figsize=(6,4))
plt.plot(DataSize, Pw_Sim_ElMod2_RMSE_Alpha , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Elimination model")
plt.plot(DataSize, Pw_Sim_PwMod2_RMSE_Alpha, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="Pairwise model")
plt.plot(DataSize, Pw_Sim_CbMod2_RMSE_Alpha, linewidth=2.5, marker='X', color='r',linestyle='-.', 
         label="Composite model")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.05, 1.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('RMSE/$\\alpha_{max}$', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title("Pairwise Data", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('PwSim_RMSE_AllModel.jpeg', format='jpeg', dpi=300 )
plt.show(block=False)

########=================================================================
######## ---- MAE for all the models and the survival ranking -----------
########=================================================================

####*********************************************************************
### ------ Elimination model --------------------------------------------
El_Sim_ElMod2_MAE_Rank = Sim_CSV_Results['El_Sim_ElMod2']['cv_pred_MAE']
El_Sim_PwMod2_MAE_Rank = Sim_CSV_Results['El_Sim_PwMod2']['cv_pred_MAE']
El_Sim_CbMod2_MAE_Rank = Sim_CSV_Results['El_Sim_CombMod2']['cv_pred_MAE']
El_Sim_CombMod2_Surv_MAE_Rank = Sim_CSV_Results['El_Sim_CombMod2']['cv_team_size_MAE']

plt.figure(figsize=(7,4))
plt.plot(DataSize, El_Sim_ElMod2_MAE_Rank , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Elimination model")
plt.plot(DataSize, El_Sim_PwMod2_MAE_Rank, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="Pairwise model")
plt.plot(DataSize, El_Sim_CbMod2_MAE_Rank, linewidth=2.5, marker='v', color='r',linestyle='-.', 
         label="Composite model")
plt.plot(DataSize, El_Sim_CombMod2_Surv_MAE_Rank, linewidth=2.5, marker='P', color='g',linestyle=':', 
         label="Survival ranking")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.55, 3.60])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)
plt.title("Elimination Data", fontweight='bold', fontsize=13)
plt.grid(True)
plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.tight_layout()
plt.savefig('ElSim_MAE_AllModel.jpeg', format='jpeg', dpi=300 )
plt.show(block=False)

####*********************************************************************
### ------ Pairwise data  ---------------------------------------
Pw_Sim_ElMod2_MAE_Rank = Sim_CSV_Results['Pw_Sim_ElMod2']['cv_pred_MAE']
Pw_Sim_PwMod2_MAE_Rank = Sim_CSV_Results['Pw_Sim_PwMod2']['cv_pred_MAE']
Pw_Sim_CbMod2_MAE_Rank = Sim_CSV_Results['Pw_Sim_CombMod2']['cv_pred_MAE']
Pw_Sim_CbMod2_Surv_MAE_Rank = Sim_CSV_Results['Pw_Sim_CombMod2']['cv_team_size_MAE']

plt.figure(figsize=(7,4))
plt.plot(DataSize, Pw_Sim_ElMod2_MAE_Rank , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Elimination model")
plt.plot(DataSize, Pw_Sim_PwMod2_MAE_Rank, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="Pairwise model")
plt.plot(DataSize, Pw_Sim_CbMod2_MAE_Rank, linewidth=2.5, marker='v', color='r',linestyle='-.', 
         label="Composite model")
plt.plot(DataSize, Pw_Sim_CbMod2_Surv_MAE_Rank, linewidth=2.5, marker='P', color='g',linestyle=':', 
         label="Survival ranking")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.55, 3.6])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('MAE', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title('Pairwise Data', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_MAE_AllModel.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)


####*********************************************************************
### ------ Spearman for various model  ----------------------------------
El_Sim_ElMod2_Spearman = Sim_CSV_Results['El_Sim_ElMod2']['cv_pred_Spearman']
El_Sim_PwMod2_Spearman = Sim_CSV_Results['El_Sim_PwMod2']['cv_pred_Spearman']
El_Sim_CbMod2_Spearman = Sim_CSV_Results['El_Sim_CombMod2']['cv_pred_Spearman']
El_Sim_CbMod2_Surv_Spearman = Sim_CSV_Results['El_Sim_CombMod2']['cv_team_size_Spearman']

plt.figure(figsize=(7,4))
plt.plot(DataSize, El_Sim_ElMod2_Spearman , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Elimination model")
plt.plot(DataSize, El_Sim_PwMod2_Spearman, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="Pairwise model")
plt.plot(DataSize, El_Sim_CbMod2_Spearman, linewidth=2.5, marker='v', color='r',linestyle='-.', 
         label="Composite model")
plt.plot(DataSize, El_Sim_CbMod2_Surv_Spearman, linewidth=2.5, marker='P', color='g',linestyle=':', 
         label="Survival ranking")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.53, 1.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('Spearman', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title("Elimination Data", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('ElSim_Spearman_AllModel.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

####*********************************************************************
### ------ Spearman for various model  ----------------------------------
Pw_Sim_ElMod2_Spearman = Sim_CSV_Results['Pw_Sim_ElMod2']['cv_pred_Spearman']
Pw_Sim_PwMod2_Spearman = Sim_CSV_Results['Pw_Sim_PwMod2']['cv_pred_Spearman']
Pw_Sim_CbMod2_Spearman = Sim_CSV_Results['Pw_Sim_CombMod2']['cv_pred_Spearman']
Pw_Sim_CbMod2_Surv_Spearman = Sim_CSV_Results['Pw_Sim_CombMod2']['cv_team_size_Spearman']

plt.figure(figsize=(7,4))
plt.plot(DataSize, Pw_Sim_ElMod2_Spearman , linewidth=2.5, marker='o', color='b',linestyle='-', 
         label="Elimination model")
plt.plot(DataSize, Pw_Sim_PwMod2_Spearman, linewidth=2.5, marker='s', color='m',linestyle='--', 
         label="Pairwise model")
plt.plot(DataSize, Pw_Sim_CbMod2_Spearman, linewidth=2.5, marker='v', color='r',linestyle='-.', 
         label="Composite model")
plt.plot(DataSize, Pw_Sim_CbMod2_Surv_Spearman, linewidth=2.5, marker='P', color='g',linestyle=':', 
         label="Survival ranking")

plt.yscale('log')
# ymin, ymax = plt.ylim()
ymin, ymax = plt.ylim([0.53, 1.0])
ticks = np.logspace(np.log10(ymin), np.log10(ymax), log_Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of matches', fontweight='bold', fontsize=13)
plt.ylabel('Spearman', fontweight='bold', fontsize=13)

plt.grid(True)

plt.legend(title='$\\alpha_{max} = 2.0$', title_fontsize=13)
plt.title("Pairwise Data", fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('PwSim_Spearman_AllModel.jpeg', format='jpeg', dpi=300 )

plt.show(block=False)

plt.show()





