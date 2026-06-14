"""
This function performs the monte carlo leave match out which was used for 
the model selection
"""

##### ==================================================================
##### ------- Importing packages needed for cross-validation -----------
##### ==================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv, LinAlgError
from tabulate import tabulate
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import re
import time
import os

####### ================================================================
####### -------- Local working directory -------------------------------
####### ================================================================
'''
To run this code, remember to change this to the working directory on your local computer.
Don't forget to change the path of the Excel folder as well
Failure to do so will render error.
'''
os.chdir('/Users/williameboannan/Documents/Programming/PUBG_GitHub_Code')


####### ================================================================
####### ------- Calling helper functions ------------------------------- 
####### ================================================================
from CompHelperFunc import (fit_logistic_cv_model,
                            generate_cv_folds,
                            comb_fit_logistic_model )


##### ==================================================================
#### --------- Importing the .csv results ------------------------------
##### ==================================================================
def get_stack_match_data():

    ## Get the folder directory
    data_folder = "/Users/williameboannan/Documents/Programming/Game_Data_Edit"
    ## Getting the .csv files in this directory
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    ## Sorting the match data in order
    csv_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))  
    # Full paths to csv files 
    csv_paths = [os.path.join(data_folder, f) for f in csv_files]
    # Read all match dataframes
    match_dfs = [pd.read_csv(fp).dropna() for fp in csv_paths]

    return match_dfs

####### ================================================================
####### ------- Defining the inputs values ----------------------------- 
####### ================================================================
NumOfTeams = 16                   ## Number of teams in the tournament 
seedNum = 125                     ## Random seed to ensure reproducibility
alpha_max = 2.0                   ## Maximum latent team strength

TrainMatches = 8                  ## Number of matches in the training set 

seedNum = 125                     ## Seed number for reproducibility
NumSplits = 100                   ## Number of Monte Carlo splits

n_splits = 6                      ## Number of cross validation folds

real_data = True                  ## Indicator for selecting a the real match data

Np = 10                          ## number of lambda domain partition

lambdas = np.exp(np.linspace(np.log(0.001), np.log(10), Np))

Models = {  "Elimination": [1.0, 0.0],
            "Pairwise":    [0.0, 1.0],
            "Composite":   [0.5, 0.5] }

###### ==========================================================
###### ----- Initializing the Storage ---------------------------
###### ==========================================================
SpearmanResults = {m: [] for m in Models}

MADResults = {m: [] for m in Models}

BestSpearmanCounts = {m: 0 for m in Models}
BestMADCounts = {m: 0 for m in Models}

rng = np.random.default_rng(seedNum)

###### ==========================================================
###### ------ Defining survival score ---------------------------
###### ==========================================================
def survival_score_one_match(df):
    matN = df.iloc[:, 11:-2].to_numpy()
    return np.mean(matN, axis=0)

###### ==========================================================
###### ------ Performing the Monte Carlo spliting ---------------
###### ==========================================================
match_dfs = get_stack_match_data()       ## Get all the csv match data

for split in range(NumSplits):

    print(f"Monte Carlo Rep {split+1}/{NumSplits}")

    ####----- Split data --------------------------
    match_idx = np.arange(len(match_dfs))

    train_idx = np.sort( rng.choice(match_idx, size=TrainMatches, replace=False) )

    valid_idx = np.sort( np.setdiff1d(match_idx, train_idx) )

    ### ------- Training and validation data -------------------

    train_df = pd.concat( [match_dfs[i] for i in train_idx], ignore_index=True )

    
    ###### --------- Validation survival score ------------
    surv_scores = [ survival_score_one_match(match_dfs[idx]) for idx in valid_idx ]

    surv_score = np.mean(np.vstack(surv_scores), axis=0)
    surv_rank = rankdata(-surv_score, method="average")

    ##### ------ CV folds ---------------------------------
    fold_assignments = generate_cv_folds(train_df, n_splits=n_splits, seedNum=1000 + split )

    SplitSpearman = {}
    SplitMAD = {}

    ######=====================================================
    ######------- Fit models ---------------------------------
    for model_name, omega in Models.items():

        cv_res = fit_logistic_cv_model( df=train_df, lambdas=lambdas, omega=omega, n_splits=n_splits, seedNum=seedNum,
                                        mu0=None, fold_assignments=fold_assignments, use_hot_start=True, real_data=real_data )

        opt_lambda = cv_res["Opt_lambda"]
        print(f'optimal lambda for {model_name} split {split} = ', opt_lambda)

        fit_res = comb_fit_logistic_model( df=train_df, lam=opt_lambda, omega=omega, NumOfTeams=NumOfTeams, seedNum=seedNum,
                                            alpha_max=alpha_max, mu0=None, real_data=real_data)

        alpha_hat = fit_res["alpha_hat"]

        

        pred_rank = rankdata(-alpha_hat, method="average")

        rho, _ = spearmanr(pred_rank, surv_rank)

        mad_rank = np.mean(np.abs(pred_rank - surv_rank))

        SpearmanResults[model_name].append(rho)
        MADResults[model_name].append(mad_rank)

        SplitSpearman[model_name] = rho
        SplitMAD[model_name] = mad_rank
    # print(SplitMAD)
    # ------------------------------------------------------
    # Best model per split
    # ------------------------------------------------------
    BestSpearmanModel = max(SplitSpearman, key=SplitSpearman.get)
    BestMADModel = min(SplitMAD, key=SplitMAD.get)

    BestSpearmanCounts[BestSpearmanModel] += 1
    BestMADCounts[BestMADModel] += 1

    # print('BestSpearmanCounts',BestSpearmanCounts)
    # print('BestMADCounts',BestMADCounts)

# ==========================================================
# FINAL SUMMARY TABLES
# ==========================================================

def summarize(results_dict, best_counts, num_splits):

    summary = {}

    for model in results_dict:

        vals = np.array(results_dict[model])

        mean_val = np.mean(vals)
        se_val = np.std(vals, ddof=1) / np.sqrt(num_splits)
        prop_best = best_counts[model] / num_splits

        summary[model] = [mean_val, se_val, prop_best]

    return summary


##### =========================================================
##### -------- Build summaries ---------------------------------
##### =========================================================
spear_summary = summarize(SpearmanResults, BestSpearmanCounts, NumSplits)
mad_summary   = summarize(MADResults, BestMADCounts, NumSplits)


##### =========================================================
##### --------- Create tables ---------------------------------

spearman_table = pd.DataFrame(spear_summary, index=["Mean", "SE", "Proportion Best"] ).round(5)

mad_table = pd.DataFrame(mad_summary, index=["Mean", "SE", "Proportion Best"] ).round(5)

# Rename model columns
spearman_table.columns = ["Elimination", "Pairwise", "Composite"]
mad_table.columns = ["Elimination", "Pairwise", "Composite"]


### ------- Add first column (Metric label) -------------

spearman_table.insert(0, "Spearman", spearman_table.index)
mad_table.insert(0, "MA Rank Diff", mad_table.index)

###### ************************************************************
###### ------- Print table ----------------------------------------
print("\nSPEARMAN TABLE")
print(tabulate(spearman_table, headers='keys', tablefmt='pretty', showindex=False))

print("\nMEAN ABSOLUTE DIFFERENCE TABLE")
print(tabulate(mad_table, headers='keys', tablefmt='pretty', showindex=False))


###### ************************************************************
###### ------- Saving the Spearman's table ------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
table = ax.table( cellText=spearman_table.values,
                  colLabels=spearman_table.columns,
                  cellLoc='center',
                  loc='center') 

table.auto_set_font_size(False)
table.set_fontsize(10)
# table.scale(2, 1.5)
plt.savefig("Speaman_Table.jpeg", dpi=300, bbox_inches='tight')
plt.show()

###### *********************************************************************
###### ------- Saving the mean absolute different table --------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
table = ax.table( cellText=mad_table.values,
                  colLabels=mad_table.columns,
                  cellLoc='center',
                  loc='center') 

table.auto_set_font_size(False)
table.set_fontsize(14)
# table.scale(2, 1.5)
plt.savefig("MAD_Table.jpeg", dpi=300, bbox_inches='tight')
plt.show()
































# # def get_stack_match_data():

# #     ## Get the folder directory
# #     data_folder = "/Users/williameboannan/Documents/Programming/Game_Data_Edit"
# #     ## Getting the .csv files in this directory
# #     csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
# #     ## Sorting the match data in order
# #     csv_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))  
# #     # Full paths to csv files 
# #     csv_paths = [os.path.join(data_folder, f) for f in csv_files]
# #     # Read all match dataframes
# #     match_dfs = [pd.read_csv(fp).dropna() for fp in csv_paths]

# #     return match_dfs


# # #==========================================================
# # # Helper: Survival ranking
# # #==========================================================
# # def get_survival_ranking(df):

# #     matN = df.iloc[:,11:-2]
# #     NumOfTeams = matN.shape[1]

# #     Surv_rank= ranking_based_on_team_size_per_encounter(NumOfTeams, matN)
    
# #     TS_rank = Surv_rank[:,1]

# #     return TS_rank

# # # ==========================================================
# # # Helper: Predicted ranking from fitted model
# # # ==========================================================
# # def get_predicted_ranking(NumOfTeams, alpha):

# #     Pred_rank = ranking_by_team_strength(NumOfTeams, alpha)

# #     alpha_rank = Pred_rank[:,1]

# #     return alpha_rank


# # # rng = np.random.default_rng()
# # # alpha = rng.uniform(-2, 2, size = 16)

# # # match_dfs = get_stack_match_data()

# # # df = match_dfs[1]
# # # alpha_rank = get_predicted_ranking(16, alpha)
# # # print(alpha_rank)

# # # # match_dfs = get_stack_match_data()
# # # # df = match_dfs[1]



# # # # surv_rank = get_survival_ranking(df)
# # # # alpha_rank = get_predicted_ranking(16, alpha)

# # # # print('Survival rank size = ', surv_rank.shape, 'Alpha rank size = ', alpha_rank.shape)
# # # # print(alpha_rank)
# # # # print(surv_rank)

# # ## ==========================================================
# # ## Monte Carlo validation
# # ## ==========================================================
# # match_dfs = get_stack_match_data()

# # NumOfTeams = 16
# # TrainMatches = 8
# # alpha_max = 2.0

# # seedNum = 123

# # NumSplits = 40
# # n_splits = 6
# # real_data = True

# # Np = 20
# # lambdas = np.logspace(-8, np.log10(4), Np)

# # Models = { "Elimination": [1.0, 0.0],
# #            "Pairwise":    [0.0, 1.0],
# #            "Composite":   [0.5, 0.5] }

# # # ==========================================================
# # # Storage
# # # ==========================================================
# # SpearmanResults = {model: [] for model in Models.keys()}
# # MAEResults      = {model: [] for model in Models.keys()}

# # BestSpearmanCounts = {model: 0 for model in Models.keys()}
# # BestMAECounts      = {model: 0 for model in Models.keys()}

# # rng = np.random.default_rng(seedNum)

# # # ==========================================================
# # # Monte Carlo loop
# # # ==========================================================
# # for split in range(NumSplits):

# #     print(f"Starting split {split+1}/{NumSplits}")

# #     # ------------------------------------------------------
# #     # Random match split
# #     # ------------------------------------------------------
# #     match_idx = np.arange(len(match_dfs))

# #     train_idx = rng.choice( match_idx, size=TrainMatches, replace=False  )

# #     valid_idx = np.setdiff1d(match_idx, train_idx)

# #     # ------------------------------------------------------
# #     # Build train and validation datasets
# #     # ------------------------------------------------------
# #     train_df = pd.concat( [match_dfs[i] for i in train_idx], ignore_index=True )

# #     valid_df = pd.concat( [match_dfs[i] for i in valid_idx], ignore_index=True )

# #     # ------------------------------------------------------
# #     # Survival ranking from validation matches
# #     # ------------------------------------------------------
# #     surv_rank = get_survival_ranking(valid_df).astype(int)

# #     SplitSpearman = {}
# #     SplitMAE = {}

# #     # ------------------------------------------------------
# #     # Loop over models
# #     # ------------------------------------------------------
# #     for model_name, omega in Models.items():

# #         # --------------------------------------------------
# #         # CV on training matches
# #         # --------------------------------------------------
# #         cv_res = fit_logistic_cv_model( df=train_df, lambdas=lambdas, omega=omega, n_splits=n_splits, seedNum=seedNum,
# #                                         mu0=None, fold_assignments=None, use_hot_start=True, real_data=real_data )

# #         opt_lambda = cv_res["Opt_lambda"] 

# #         # --------------------------------------------------
# #         # Fit model on validation matches
# #         # --------------------------------------------------
# #         fit_res = comb_fit_logistic_model( df=train_df, lam=opt_lambda, omega=omega, NumOfTeams=NumOfTeams,
# #                                             seedNum=seedNum, alpha_max=alpha_max, mu0=None, real_data=True )

# #         alpha_hat = fit_res["alpha_hat"]

# #         # --------------------------------------------------
# #         # Predicted ranking
# #         # --------------------------------------------------
# #         pred_rank = get_predicted_ranking( NumOfTeams, alpha_hat ).astype(int)

# #         # --------------------------------------------------
# #         # Convert rankings to positions
# #         # --------------------------------------------------
# #         surv_pos = np.empty(NumOfTeams, dtype=int)
# #         pred_pos = np.empty(NumOfTeams, dtype=int)

# #         surv_pos[surv_rank - 1] = np.arange(NumOfTeams)
# #         pred_pos[pred_rank - 1] = np.arange(NumOfTeams)

# #         # --------------------------------------------------
# #         # Metrics
# #         # --------------------------------------------------
# #         mae = np.mean(np.abs(surv_pos - pred_pos))

# #         rho, _ = spearmanr( surv_pos, pred_pos )

# #         # Store results
# #         SpearmanResults[model_name].append(rho)
# #         MAEResults[model_name].append(mae)

# #         SplitSpearman[model_name] = rho
# #         SplitMAE[model_name] = mae

# #     # ------------------------------------------------------
# #     # Best model for this split
# #     # ------------------------------------------------------
# #     BestSpearmanModel = max( SplitSpearman, key=SplitSpearman.get )

# #     BestMAEModel = min( SplitMAE, key=SplitMAE.get )

# #     BestSpearmanCounts[BestSpearmanModel] += 1
# #     BestMAECounts[BestMAEModel] += 1


# # # ==========================================================
# # # TABLE 1 : SPEARMAN
# # # ==========================================================
# # print("\n")
# # print("=" * 70)
# # print("SPEARMAN CORRELATION")
# # print("=" * 70)

# # for model in Models.keys():

# #     MeanVal = np.mean( SpearmanResults[model] )

# #     StdErr = ( np.std( SpearmanResults[model], ddof=1 )/ np.sqrt(len(SpearmanResults[model])) )

# #     PropBest = ( BestSpearmanCounts[model]/ NumSplits )

# #     print( f"{model:12s}"
# #            f" Mean={MeanVal:.4f}"
# #            f" SE={StdErr:.4f}"
# #            f" PropBest={PropBest:.4f}" )


# # # ==========================================================
# # # TABLE 2 : MAE
# # # ==========================================================
# # print("\n")
# # print("=" * 70)
# # print("MEAN ABSOLUTE RANK DIFFERENCE")
# # print("=" * 70)

# # for model in Models.keys():

# #     MeanVal = np.mean( MAEResults[model] )

# #     StdErr = ( np.std( MAEResults[model], ddof=1 )/ np.sqrt(len(MAEResults[model])))

# #     PropBest = (
# #         BestMAECounts[model]
# #         /
# #         NumSplits
# #     )

# #     print(
# #         f"{model:12s}"
# #         f" Mean={MeanVal:.4f}"
# #         f"  SE={StdErr:.4f}"
# #         f"  PropBest={PropBest:.4f}"
# #     )




