'''
This function uses only the real match data
The function:
1. import all the csv real match data and arrange them (Remember to specity the data directory)
2. Starting with a single match, stack of 2 matches, ..., the stack of all the matches,
   - the function performs a cross validation for each stack
   - fit both the unpenalized and penalized model to obtained optimal parameter values
3. plot the sensitivity curve for beta and gamma against the data size
'''

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

####### ================================================================
####### ------- Calling helper functions ------------------------------- 
####### ================================================================
from CompHelperFunc import (fit_logistic_cv_model,
                            comb_fit_logistic_model)

########### ==============================================================
#### --------- This sunction get the stuck data---------------------------
def get_stack_match_data():
    '''
    - This function extract the various match data set into a folder 
    - Sort these files (match data) using the numbers in the file name (eg match 1, match 2 etc)
    '''
    
    
    ## Get the folder directory: this is the folder where the match data is store
    data_folder = "/Users/williameboannan/Documents/Programming/PUBG_GitHub_Code/Game_Data"
    ## Getting the .csv files in this directory
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    ## Sorting the match data in order
    csv_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))  
    # Full paths to csv files 
    csv_paths = [os.path.join(data_folder, f) for f in csv_files]
    # Read all match dataframes
    match_df = [pd.read_csv(fp).dropna() for fp in csv_paths]


    return match_df


####### ================================================================
####### ------- Defining the inputs values ----------------------------- 
####### ================================================================

StartTime= time.perf_counter()     ## start the timer to check the simulation time

"""
When dealing with real data, these inputs are not needed, 
however, because they are inputs to the function to the function 
that fit the logistic regression model and also perform cross validation,
they need to be specified.
"""
NumOfTeams = 16                   ## Number of teams in the tournament 
n0 = 4                            ## Team size/game format (solo=1, duo = 2, squad =4)
seedNum = 125                     ## Random seed to ensure reproducibility
alpha_max = 2.0                   ## Maximum latent team strength

elim_omega = 0.50                  ## weight assigned to the elimination model
pw_omega = 1- elim_omega          ## weight assigned to the pairwise model
omega = [elim_omega , pw_omega]   ## weight vector  

n_splits = 6                      ## number of split for cross validation

Np = 250                          ## number of partition for the lambda domain
lam0 = 0.0                        ## lambda value for unpenalized model
marker_spacing = int(np.ceil(Np/20))         ## marker spacing when ploting 

## lambdas = np.linspace(0,10, Np)           ## used for uniform partition of the lambda domain
lambdas = np.logspace(-8, np.log10(30), Np)  ## used for logarithmic partition of the lambda domain

real_data = True                  ## selecting either real or simulated match data
elimination_data= True             ## selecting either elimination or pairwise data

if elim_omega == 1.0:
    ModelType = 'El_Model'
elif elim_omega== 0.0:
    ModelType = 'Pw_Model'
else:
    ModelType = 'Comb_Model'

####### ================================================================
####### ------- Performing the simulation ------------------------------ 

match_dfs = get_stack_match_data()       ## Getting all the real match data

alpha_results = {}         ## Dictionary to store the alpha values 
opt_lam_vals = []          ## array to store the optimal lambda values 
CV_beta = []               ## array to store the optimal cross validation beta values 
Unpen_beta = []            ## array to store the optimal unpenalized beta values 
CV_gamma = []              ## array to store the optimal cross validation beta values 
Unpen_gamma = []           ## array to store the optimal unpenalized beta values
PerfData = []


for k in range(len(match_dfs)):
    '''
    Here, k = j means running the simulation for a stack of j+1 matches,
    '''
    
    ## ------- Get a stack of k+1 matches --------------------------
    df = pd.concat(match_dfs[:k+1], ignore_index=True)

    # print(f"\nRunning analysis for {k+1} out of {len(match_dfs)} matches")
    print(f"Starting analysis for a stack of {k+1}/{len(match_dfs)} matches" )
    
    ## --- perform Cross-validation to get optimal lambda ------
    cv_res = fit_logistic_cv_model(df, lambdas, omega, n_splits=n_splits, seedNum=seedNum, mu0=None, 
                                   fold_assignments=None, use_hot_start=True, real_data= real_data)
    
    opt_lambda = cv_res['Opt_lambda']     ## optimal lambda from cross validation
    
    opt_lam_vals.append(opt_lambda)

    print(f'Optimal lambda for a stack of {k+1} matches is', opt_lambda)

    ## -------Fit the the unpenalized model  ------------------
    fit_unpen = comb_fit_logistic_model(df, lam=lam0, omega=omega, NumOfTeams=NumOfTeams, 
                                        seedNum=seedNum, alpha_max= alpha_max, mu0=None, real_data= real_data)

    Unpen_beta.append(fit_unpen['beta_hat'])
    Unpen_gamma.append(fit_unpen['gamma_hat'])

    ## -------Fit the the penalized model  ------------------
    fit_pen = comb_fit_logistic_model(df, lam=opt_lambda, omega=omega, NumOfTeams=NumOfTeams, 
                                        seedNum=seedNum, alpha_max= alpha_max, mu0=None, real_data= real_data)
    
    CV_beta.append(fit_pen['beta_hat'])
    CV_gamma.append(fit_pen['gamma_hat'])

    ## -------------------------------------------------
    ## Ignore the alpha for the game setting when its involved in the elimination
    if len(fit_unpen['alpha_hat']) > NumOfTeams:
        alpha_results[f"unpen_stack_{k+1}"] = fit_unpen['alpha_hat'][:-1]
        alpha_results[f"pen_stack_{k+1}"]   = fit_pen['alpha_hat'][:-1]
    else: 
        alpha_results[f"unpen_stack_{k+1}"] = fit_unpen['alpha_hat']
        alpha_results[f"pen_stack_{k+1}"]   = fit_pen['alpha_hat']


    PerfData.append( {  "Stack_matches": k+1,
                        'optimal lambda': opt_lambda,
                        "unpen_beta": fit_unpen['beta_hat'],
                        "cv_beta": fit_pen['beta_hat'],
                        "unpen_gamma": fit_unpen['gamma_hat'],
                        "cv_gamma": fit_pen['gamma_hat'] })

## ---- Saving alpha and optimal lambda values into a pandas DataFrame and CSV file
alpha_df = pd.DataFrame(alpha_results)
alpha_df.to_csv(f"{ModelType}_Alphas.csv", index=False)

## --- Saving preformance result ----------
Performance = pd.DataFrame(PerfData)
Performance.to_csv(f"{ModelType}_PerfData.csv", index=False)

##### =======================================================
EndTime = time.perf_counter()
print(f'Simulation Time = {EndTime - StartTime:.5f} seconds')   ## Print out the simulation time

#######==================================================
Match_num = np.arange(1, len(match_dfs)+1)
Calibrations = 10
### -------- Beta curve--------------------------------
plt.figure(figsize=(7,4))
plt.plot(Match_num, Unpen_beta, linewidth=2.5, marker='o', color='b',linestyle='-', label="unpen")
plt.plot(Match_num, CV_beta, linewidth=2.5, marker='s', color='m',linestyle='--', label="cv")

ymin, ymax = plt.ylim()
ticks = np.linspace(ymin, ymax, Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])

plt.xlabel('Number of matches', fontweight='bold',fontsize=13)
plt.ylabel('Optimal $\\beta$', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Beta_Curve.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

### -------- Gamma curve--------------------------------
plt.figure(figsize=(7,4))
plt.plot(Match_num, Unpen_gamma, linewidth=2.5, marker='o', color='b',linestyle='--', label="unpen")
plt.plot(Match_num, CV_gamma, linewidth=2.5, marker='s', color='m',linestyle=':', label="cv")

ymin, ymax = plt.ylim()
ticks = np.linspace(ymin, ymax, Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])

plt.xlabel('Number of matches', fontweight='bold',fontsize=14)
plt.ylabel('Optimal $\\gamma$', fontweight='bold',fontsize=14)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Gamma_Curve.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

### -------- lambda curve--------------------------------
plt.figure(figsize=(8,4))
plt.plot(Match_num, opt_lam_vals, linewidth=2.5, marker='o', color='b',linestyle='-.')

ymin, ymax = plt.ylim()
ticks = np.linspace(ymin, ymax, Calibrations)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])

plt.xlabel('Number of matches', fontweight='bold',fontsize=13)
plt.ylabel('optimal $\\lambda$', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.savefig(f'Lambda.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

plt.show()
