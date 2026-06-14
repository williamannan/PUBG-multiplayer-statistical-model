'''
The function performs a simulation study to access the performance of the proposed models.

The function:
1. Generate true latent teaam strength (alpha)
2. Simulate a series a stack match data. For each stack of data:
   - we perform a cross validation to get optimal lambda value
   - fit both the penalized and unpenalized model to the stack match data
   - record the the various performance metrics outlined in the helper function
3. The output is saved into a csv file which can be used later
4. Some initial plot are also generated

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
import matplotlib.pyplot as plt
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
####### ------- Calling helper functions from the  --------------------- 
####### ================================================================
'''
Refer to the "CompHelperFunc" for details of each of these
'''
from CompHelperFunc import (comb_fit_logistic_model, 
                            fit_logistic_cv_model,
                            combined_match_data,
                            generate_alpha )

####### ================================================================
####### ------- Defining the inputs values ----------------------------- 
####### ================================================================

StartTime = time.perf_counter()

NumOfTeams = 16                   ## Number of teams in the tournament 
n0 = 4                            ## Team size/game format (solo=1, duo = 2, squad =4)
seedNum = 125                     ## Random seed to ensure reproducibility
alpha_max = 2.0                   ## Maximum latent team strength

elim_omega = 0.50                  ## weight assigned to the elimination model
pw_omega = 1- elim_omega          ## weight assigned to the pairwise model
omega = [elim_omega , pw_omega]   ## weight vector  

n_splits = 6                      ## number of split for cross validation

Np = 20                          ## number of partition for the lambda domain
lam0 = 0.0                        ## lambda value for unpenalized model
marker_spacing = int(np.ceil(Np/20))         ## marker spacing when ploting 

## lambdas = np.linspace(0,10, Np)           ## used for uniform partition of the lambda domain
lambdas = np.logspace(-8, np.log10(10), Np)  ## used for logarithmic partition of the lambda domain

real_data = False                  ## selecting either real or simulated match data
elimination_data= True             ## selecting either elimination or pairwise data

max_num_of_stack_data = 12         ## Maxinum number of match data to be stacked together
max_repeat_seq = 10                ## maximum number of repeated simulation

####-------- Labeling ------------------------------
if elim_omega == 0.0:
    model_type = "Pw_mod"
elif elim_omega == 1.0:
    model_type = "El_Mod"
else:
    model_type = "Cb_Mod"

if elimination_data is True:
    Data_type = 'ElData'
else:
    Data_type = 'PwData'

label = f"{Data_type}_{model_type}_{alpha_max}"

####### ================================================================
####### ------- Performing simulation study ---------------------------- 

AllPerformData = []     ## Initialize a matrix to store all the results

optimal_lambdas = {}

alpha = generate_alpha(NumOfTeams,alpha_max, seedNum)


for DataSize in range(1, max_num_of_stack_data + 1):
    ''' 
    In the first for loop, we:
    - Select the number of matches to be stack together.

    In the second for loop, we:
    - Repeatedly generate a number of stack data
    - Run a cross validation for each stack match data generated 
    - fit both the penalized and unpenalized model
    - Record performance metrics such as RMSE, MAE, Spearman, etc

    At the end of the loops, compute the average of the recorded values.
    save the result in csv format.
    '''

    print(f'Starting analysis for {DataSize}/{max_num_of_stack_data} stack of matches')

    ## -- Initialize vectrs for beta and gamma --------------------
    unpen_beta_list = []
    cv_beta_list = []
    unpen_gamma_list = []
    cv_gamma_list = []

    ## -- Initialize vectrs for the repeated unpenalize model fit --------
    unpen_pred_rmse_list = []
    unpen_pred_mae_list = []
    unpen_pred_spearman_list = []
    unpen_perfect_pred_list = []
    unpen_wrong_pred_list = []
    
    unpen_ts_rmse_list = []
    unpen_ts_mae_list = []
    unpen_ts_spearman_list = []
    unpen_ts_perfect_pred = []
    unpen_ts_wrong_pred  = []

    unpen_comb_rmse_list = []
    unpen_comb_mae_list = []
    unpen_comb_spearman_list = []
    unpen_comb_perfect_pred = []
    unpen_comb_wrong_pred = []

    unpen_alpha_rmse_list = []
    unpen_alpha_mae_list = []
    unpen_opt_nll_list = []

    ##-- Initialized vectrs for the repeated unpenalize model fit ---
    cv_pred_rmse_list = []
    cv_pred_mae_list = []
    cv_pred_spearman_list = []
    cv_perfect_pred_list = []
    cv_wrong_pred_list = []
    
    cv_ts_rmse_list = []
    cv_ts_mae_list = []
    cv_ts_spearman_list = []
    cv_ts_perfect_pred = []
    cv_ts_wrong_pred  = []

    cv_comb_rmse_list = []
    cv_comb_mae_list = []
    cv_comb_spearman_list = []
    cv_comb_perfect_pred = []
    cv_comb_wrong_pred = []

    cv_alpha_rmse_list = []
    cv_alpha_mae_list = []
    cv_opt_nll_list = []

    Loc_lambdas = []

    for k_repeat in range(max_repeat_seq):
        
        df_stack = combined_match_data(alpha, n0, num_of_stack_data=DataSize, elimination_data=elimination_data, 
                                       data_rand_state=None)
        
        cross_val = fit_logistic_cv_model(df_stack, lambdas, omega, n_splits=n_splits, seedNum=seedNum,
                                          mu0=None, fold_assignments=None, use_hot_start=True, real_data =real_data)
        
        opt_lambda = cross_val['Opt_lambda']

        Loc_lambdas.append(opt_lambda)


        unpen_res = comb_fit_logistic_model(df= df_stack, lam=0.0, omega= omega, NumOfTeams= NumOfTeams,
                                            seedNum= seedNum, alpha_max=alpha_max, mu0=None, real_data= real_data)
        
        
        cv_res = comb_fit_logistic_model(df= df_stack, lam=opt_lambda, omega= omega, NumOfTeams= NumOfTeams,
                                            seedNum= seedNum, alpha_max=alpha_max, mu0=None, real_data= real_data)
        
        ## ----- Sensitivity result from CV and unpenalized model ------
        unpen_beta_list.append(unpen_res['beta_hat'])
        cv_beta_list.append(cv_res['beta_hat'])
        unpen_gamma_list.append(unpen_res['gamma_hat'])
        cv_gamma_list.append(cv_res['gamma_hat'])

        ## ----- Results from the unpenalized model ------
        unpen_pred_rmse_list.append(unpen_res['Predicted RMSE'])
        unpen_pred_mae_list.append(unpen_res['Predicted MAE'])
        unpen_pred_spearman_list.append(unpen_res['Predicted Spearmans'])
        unpen_perfect_pred_list.append(unpen_res['Perfect prediction'])
        unpen_wrong_pred_list.append(unpen_res['Wrong prediction'])

        unpen_ts_rmse_list.append(unpen_res['Team size RMSE'])
        unpen_ts_mae_list.append(unpen_res['Team size MAE'])
        unpen_ts_spearman_list.append(unpen_res['Team size Spearmans'])
        unpen_ts_perfect_pred.append(unpen_res['Team size perfect_pred'])
        unpen_ts_wrong_pred.append(unpen_res['Team size wrong_pred'])

        unpen_comb_rmse_list.append(unpen_res['Combined RMSE'])
        unpen_comb_mae_list.append(unpen_res['Combined MAE'])
        unpen_comb_spearman_list.append(unpen_res['Combined Spearmans'])
        unpen_comb_perfect_pred.append(unpen_res['Combined perfect_pred'])
        unpen_comb_wrong_pred.append(unpen_res['Combined worng_pred'])

        unpen_alpha_rmse_list.append(unpen_res['RMSE between alpha'])
        unpen_alpha_mae_list.append(unpen_res['MAE between alpha'])
        unpen_opt_nll_list.append(unpen_res['Opt_lik_est'])

        ## ----- Results from the penalized model ------
        cv_pred_rmse_list.append(cv_res['Predicted RMSE'])
        cv_pred_mae_list.append(cv_res['Predicted MAE'])
        cv_pred_spearman_list.append(cv_res['Predicted Spearmans'])
        cv_perfect_pred_list.append(cv_res['Perfect prediction'])
        cv_wrong_pred_list.append(cv_res['Wrong prediction'])

        cv_ts_rmse_list.append(cv_res['Team size RMSE'])
        cv_ts_mae_list.append(cv_res['Team size MAE'])
        cv_ts_spearman_list.append(cv_res['Team size Spearmans'])
        cv_ts_perfect_pred.append(cv_res['Team size perfect_pred'])
        cv_ts_wrong_pred.append(cv_res['Team size wrong_pred'])

        cv_comb_rmse_list.append(cv_res['Combined RMSE'])
        cv_comb_mae_list.append(cv_res['Combined MAE'])
        cv_comb_spearman_list.append(cv_res['Combined Spearmans'])
        cv_comb_perfect_pred.append(cv_res['Combined perfect_pred'])
        cv_comb_wrong_pred.append(cv_res['Combined worng_pred'])

        cv_alpha_rmse_list.append(cv_res['RMSE between alpha'])
        cv_alpha_mae_list.append(cv_res['MAE between alpha'])
        cv_opt_nll_list.append(cv_res['Opt_lik_est'])


    # Average over all the repeated simulations
    AllPerformData.append({ 'Num_Stacked_Data': DataSize,
                            'unpen_beta': np.mean(unpen_beta_list),
                            'CV_beta': np.mean(cv_beta_list),
                            'unpen_gamma': np.mean(unpen_gamma_list),
                            'CV_gamma': np.mean(cv_gamma_list),

                            'unpen_pred_RMSE': np.mean(unpen_pred_rmse_list),
                            'cv_pred_RMSE': np.mean(cv_pred_rmse_list),
                            'unpen_pred_MAE': np.mean(unpen_pred_mae_list),
                            'cv_pred_MAE': np.mean(cv_pred_mae_list),
                            'unpen_pred_Spearman': np.mean(unpen_pred_spearman_list),
                            'cv_pred_Spearman': np.mean(cv_pred_spearman_list),
                            'unpen_Correct_pred': np.mean(unpen_perfect_pred_list),
                            'cv_Correct_pred': np.mean(cv_perfect_pred_list),
                            'unpen_wrong_pred': np.mean(unpen_wrong_pred_list),
                            'cv_wrong_pred': np.mean(cv_wrong_pred_list),

                            'unpen_team_size_RMSE':np.mean(unpen_ts_rmse_list),
                            'cv_team_size_RMSE':np.mean(cv_ts_rmse_list),
                            'unpen_team_size_MAE':np.mean(unpen_ts_mae_list),
                            'cv_team_size_MAE':np.mean(cv_ts_mae_list),
                            'unpen_team_size_Spearman':np.mean(unpen_ts_spearman_list),
                            'cv_team_size_Spearman':np.mean(cv_ts_spearman_list),
                            'unpen_team_size_correct_pred':np.mean(unpen_ts_perfect_pred),
                            'cv_team_size_correct_pred':np.mean(cv_ts_perfect_pred),
                            'unpen_team_size_wrong_pred':np.mean(unpen_ts_wrong_pred),
                            'cv_team_size_wrong_pred':np.mean(cv_ts_wrong_pred),

                            'unpen_combine_RMSE':np.mean(unpen_comb_rmse_list),
                            'cv_combine_RMSE':np.mean(cv_comb_rmse_list),
                            'unpen_combine_MAE':np.mean(unpen_comb_mae_list),
                            'cv_combine_MAE':np.mean(cv_comb_mae_list),
                            'unpen_combine_Spearman':np.mean(unpen_comb_spearman_list),
                            'cv_combine_Spearman':np.mean(cv_comb_spearman_list),
                            'unpen_combine_correct_pred':np.mean(unpen_comb_perfect_pred),
                            'cv_combine_correct_pred':np.mean(cv_comb_perfect_pred),
                            'unpen_combine_incorrect_pred':np.mean(unpen_comb_wrong_pred),
                            'cv_combine_incorrect_pred':np.mean(cv_comb_wrong_pred),

                            'unpen_RMSE_between_alpha':np.mean(unpen_alpha_rmse_list),
                            'cv_RMSE_between_alpha':np.mean(cv_alpha_rmse_list),
                            'unpen_MAE_between_alpha':np.mean(unpen_alpha_mae_list),
                            'cv_MAE_between_alpha':np.mean(cv_alpha_mae_list),
                            'unpen_Opt_lik_est': np.mean(unpen_opt_nll_list),
                            'cv_Opt_lik_est': np.mean(cv_opt_nll_list)                         
                            })

    # optimal_lambdas.append({f'Stuck_{k_repeat}': Loc_lambdas})
    optimal_lambdas[f'Stack_{DataSize}'] = Loc_lambdas

    print(f'Completed analysis for {DataSize}/{max_num_of_stack_data} stack of matches')
## ******* -- Saving optimal lambda values to csv file --********
lambda_values = pd.DataFrame(optimal_lambdas)
lambda_values.to_csv(f"Lamb_{label}.csv", index=False)

## ******* -- Saving performance  metrics to csv file --********
Avg_Values = pd.DataFrame(AllPerformData)
Avg_Values.to_csv(f"Perf_{label}.csv", index=False)

## ******* -- Saving input parameters to csv file --********
params = {  "NumOfTeams": NumOfTeams,
            "Team size": n0,
            "maximum strength": alpha_max,
            "max_stack_data": max_num_of_stack_data,
            "Number of repeatition": max_repeat_seq,
            "elimination weight": elim_omega,
            "K-fold": n_splits, 
            "lambda partition": Np}

pd.DataFrame([params]).to_csv(f"Par_{label}.csv", index=False)

EndTime = time.perf_counter()

print(f"Elapse time : {EndTime - StartTime:.5f} seconds")


######## ========================================================
######## ---- Making some few plots -----------------------------
######## ========================================================

datasize = Avg_Values['Num_Stacked_Data']

## -------- RMSE plot --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_pred_RMSE'],linewidth=2.5, marker='*', color='b',linestyle='--', 
         label="unpen_RMSE_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['cv_pred_RMSE'],linewidth=2.5, marker='s', color='g',linestyle='-.', 
         label="cv_RMSE_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['unpen_team_size_RMSE'],linewidth=2.5, marker='o', color='r',linestyle=':', 
         label="unpen_RMSE_team_size_rank")
plt.plot(datasize, Avg_Values['cv_team_size_RMSE'],linewidth=2.5, marker='D', color='m',linestyle='-', 
         label="cv_RMSE_team_size_rank")
plt.plot(datasize, Avg_Values['unpen_combine_RMSE'],linewidth=2.5, marker='^', color='c',linestyle=':', 
         label="unpen_RMSE_combine_rank")
plt.plot(datasize, Avg_Values['cv_combine_RMSE'],linewidth=2.5, marker='X', color='k',linestyle='-', 
         label="cv_RMSE_combine_rank")

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('RMSE', fontweight='bold',fontsize=13)
plt.title(f'RMSE for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'RMSE_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- MAE plot --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_pred_MAE'],linewidth=2.5, marker='*', color='b',linestyle='--', 
         label="unpen_MAE_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['cv_pred_MAE'],linewidth=2.5, marker='s', color='g',linestyle='-.', 
         label="cv_MAE_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['unpen_team_size_MAE'],linewidth=2.5, marker='o', color='r',linestyle=':', 
         label="unpen_MAE_team_size_rank")
plt.plot(datasize, Avg_Values['cv_team_size_MAE'],linewidth=2.5, marker='D', color='m',linestyle='-', 
         label="cv_MAE_team_size_rank")
plt.plot(datasize, Avg_Values['unpen_combine_MAE'],linewidth=2.5, marker='^', color='c',linestyle=':', 
         label="unpen_MAE_combine_rank")
plt.plot(datasize, Avg_Values['cv_combine_MAE'],linewidth=2.5, marker='X', color='k',linestyle='-', 
         label="cv_MAE_combine_rank")

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('MAE', fontweight='bold',fontsize=13)
plt.title(f'MAE for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'MAE_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- Spearman's plot --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_pred_Spearman'],linewidth=2.5, marker='*', color='b',linestyle='--', 
         label="unpen_Spearman_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['cv_pred_Spearman'],linewidth=2.5, marker='s', color='g',linestyle='-.', 
         label="cv_Spearman_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['unpen_team_size_Spearman'],linewidth=2.5, marker='o', color='r',linestyle=':', 
         label="unpen_Spearman_team_size_rank")
plt.plot(datasize, Avg_Values['cv_team_size_Spearman'],linewidth=2.5, marker='D', color='m',linestyle='-', 
         label="cv_Spearman_team_size_rank")
plt.plot(datasize, Avg_Values['unpen_combine_Spearman'],linewidth=2.5, marker='^', color='c',linestyle=':', 
         label="unpen_Spearman_combine_rank")
plt.plot(datasize, Avg_Values['cv_combine_Spearman'],linewidth=2.5, marker='X', color='k',linestyle='-', 
         label="cv_Spearman_combine_rank")

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('Spearman', fontweight='bold',fontsize=13)
plt.title(f'Spearman for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Spearman_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- Correct Prediction plot --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_Correct_pred'],linewidth=2.5, marker='*', color='b',linestyle='--', 
         label="unpen_correct_pred_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['cv_Correct_pred'],linewidth=2.5, marker='s', color='g',linestyle='-.', 
         label="cv_correct_pred_$\\alpha$_rank")
plt.plot(datasize, Avg_Values['unpen_team_size_correct_pred'],linewidth=2.5, marker='o', color='r',linestyle=':', 
         label="unpen_correct_pred_team_size_rank")
plt.plot(datasize, Avg_Values['cv_team_size_correct_pred'],linewidth=2.5, marker='D', color='m',linestyle='-', 
         label="cv_correct_pred_team_size_rank")
plt.plot(datasize, Avg_Values['unpen_combine_correct_pred'],linewidth=2.5, marker='^', color='c',linestyle=':', 
         label="unpen_correct_pred_combine_rank")
plt.plot(datasize, Avg_Values['cv_combine_correct_pred'],linewidth=2.5, marker='X', color='k',linestyle='-', 
         label="cv_correct_pred_combine_rank")

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('Correct_pred', fontweight='bold',fontsize=13)
plt.title(f'Correct_pred for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Correct_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- RMSE between alpha --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_RMSE_between_alpha'],linewidth=2.5, marker='*', color='b',linestyle='--', 
         label="unpen_RMSE_btn_$\\alpha$")
plt.plot(datasize, Avg_Values['cv_RMSE_between_alpha'],linewidth=2.5, marker='s', color='g',linestyle='-.', 
         label="cv_RMSE_btn_$\\alpha$")

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('RMSE', fontweight='bold',fontsize=13)
plt.yscale('log')
plt.title(f'RMSE between $\\alpha$ for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Alp_RMSE_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- MAE between --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_MAE_between_alpha'],linewidth=2.5, marker='*', color='r',linestyle=':', 
         label="unpen_MAE_btn_$\\alpha$")
plt.plot(datasize, Avg_Values['cv_MAE_between_alpha'],linewidth=2.5, marker='s', color='m',linestyle='-', 
         label="cv_MAE_btn_$\\alpha$")

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('MAE', fontweight='bold',fontsize=13)
plt.yscale('log')
plt.title(f'MAE between $\\alpha$ for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Alp_MAE_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- NLL plot --------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_Opt_lik_est'],linewidth=2.5, marker='*', color='b',linestyle='--', 
         label="unpen_NLL")
plt.plot(datasize, Avg_Values['cv_Opt_lik_est'],linewidth=2.5, marker='s', color='r',linestyle='-.', 
         label="cv_NLL")


plt.xlabel('Number of stack data', fontweight='bold',fontsize=12)
plt.ylabel('NLL estimate', fontweight='bold',fontsize=12)
plt.title(f'NLL estimate for {model_type} Averaged over {max_repeat_seq} simulations', fontweight='bold',fontsize=12)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'NLL_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

### -------- Beta curve--------------------------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_beta'], linewidth=2.5, marker='o', color='b',linestyle='-', label="unpen_beta")
plt.plot(datasize, Avg_Values['CV_beta'], linewidth=2.5, marker='s', color='m',linestyle='--', label="cv_beta")

ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack matches', fontweight='bold',fontsize=13)
plt.ylabel('Optimal $\\beta$', fontweight='bold',fontsize=13)
plt.title(f'$\\beta$ profile for {label}', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Beta_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

### -------- Gamma curve--------------------------------
plt.figure(figsize=(8,4))
plt.plot(datasize, Avg_Values['unpen_gamma'], linewidth=2.5, marker='o', color='b',linestyle='-', label="unpen_beta")
plt.plot(datasize, Avg_Values['CV_gamma'], linewidth=2.5, marker='s', color='m',linestyle='--', label="cv_beta")

ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack matches', fontweight='bold',fontsize=13)
plt.ylabel('Optimal $\\gamma$', fontweight='bold',fontsize=13)
plt.title(f'$\\gamma$ profile for {label}', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.legend()
plt.savefig(f'Beta_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

## -------- Optimal Lambda plot --------------
lam_means = lambda_values.mean(axis=0)
lam_std = lambda_values.std(axis=0)
lam_se = lam_std/np.sqrt(max_num_of_stack_data)

plt.figure(figsize=(8,4))
plt.errorbar(datasize, lam_means, yerr=lam_se, fmt='*-', capsize=3)

# plt.xscale('log')
ymin, ymax = plt.ylim()
ticks = np.linspace(ymin,ymax, 8)
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
plt.minorticks_off()

plt.xlabel('Number of stack data', fontweight='bold',fontsize=13)
plt.ylabel('optimal lambda value', fontweight='bold',fontsize=13)
plt.title(f'optimal lambda value for {model_type}', fontweight='bold',fontsize=13)
plt.grid(True)
plt.tight_layout()       # adjust spacing
plt.savefig(f'Opt_lamb_{label}.jpeg', format='jpeg', dpi=300)      # dpi=300 for high resolution
plt.show(block=False)

plt.show()



