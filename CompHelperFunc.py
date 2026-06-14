'''
This funtion file contain all the helper functions needed to run the entire simulations.

The functions in this file include those that:
1. Convert any given data into numpy array
2. Generate team strengths (underlying alpha values for the teams)
3. Simulate match data using both the elimnination and pairwise model
4. Define various ranking schemes (based on team strength, team sizes and the combination of the two)
5. Defines various vectors and matrices needed to define likelihood estimate and the gradient 
6. Fit logistic regression model to obtained optimal parameter values 
7. Perform cross validation
'''

####### ================================================================
####### ----- Packages needed for running the simulation ---------------
####### ================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv, LinAlgError
from tabulate import tabulate
from scipy.stats import spearmanr, rankdata
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import time
import os

####### ================================================================
####### -------- Local working directory -------------------------------
'''
To run this code, remember to change this to the working directory on your local computer.
Failure to do so will render error.
'''
os.chdir('/Users/williameboannan/Documents/Programming/PUBG_GitHub_Code')


####### ================================================================
####### --- Converting any given data into numpy array -----------------
####### ================================================================
def to_numpy(x):
    """ This function safely converts a pandas DataFrame/Series to a NumPy array.
    If x is already a NumPy array, returns it unchanged """

    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    
    return np.asarray(x)

####### ================================================================
####### --- Generating alpha values (true team strength) ---------------
####### ================================================================
def generate_alpha(NumOfTeams, alpha_max, seedNum):
    """
    Generate latent team strength parameters.

    Parameters
    ----------
    NumOfTeams : int ( Number of teams in the tournament).
    alpha_max : float ( Maximum absolute value of the team strengths).
    seedNum : int ( Random seed used for reproducibility).

    Returns
    -------
    alpha : ndarray (Vector of team strengths satisfying
        -alpha_max <= alpha_j <= alpha_max for all teams).
    """

    NumOfTeams = int(NumOfTeams)    ## Ensure NumOfTeams is integer
    seedNum = int(seedNum)          ## Ensure seedNum is integer
    alpha_max = float(alpha_max)    ## Ensure alpha_max is float

    np.random.seed(seedNum)         ## set random seed for reproducibility

    ## --- Generate alpha values, recenter and round to 5 decimal place ----
    alpha = alpha_max*np.tanh(np.random.randn(NumOfTeams))   
    alpha -= np.mean(alpha)         
    alpha =  np.round(alpha, 5)   

    ## Saving alpha values to a csv file to be used later when needed
    Strength = pd.DataFrame(alpha, columns=["alpha"]) 
    Strength.to_csv("Sim_Alpha.csv", index=False)

    return alpha

###### =============================================================
###### -------- Simulating pairwise match data ---------------------
###### =============================================================
def simulate_pairwise_competition(alpha, n0, data_rand_state=None):
    """ 
    Simulate match data from the pairwise model

    Inputs
    -----------
    alpha           : ndarray of team strengths
    n0              : int (initial team size, Solo =1, Duo = 2, Squad =4)
    data_rand_state : Set to None to to generate different data on each run

    Output 
    ---------
    Pw_match_data: Pandad DataFrame of sequence of team sizes 

    Refer to the "STATISTICAL MODEL FOR MULTI-PLAYER GAME DYNAMICS" for details of the model
    """
    
    alpha = to_numpy(alpha).ravel()    ## an array of letent strength
    n0 = int(n0)                       ## Initial team sizes
    gamma = 1.50                       ## Sensitivity parameter obtained from the real data

    if data_rand_state is not None:
        data_rand_state = int(data_rand_state)

    N = len(alpha)                    ## The length of alpha (number of teams)

    ## setting random_state = None generate different data anytime the code is run
    rng = np.random.default_rng(data_rand_state)

    ## Initializing team sizes before event 1
    n = np.full(N, n0, dtype=int)    

    match_records = []                ## Initialize matrix to store match data

    k = 1

    while np.sum(n > 0) > 1:
        ### --- Select two alive teams ---
        '''
        - Extract the indicies of the active/alive teams (team with nonzero sizes).
        - Randomly choose 2 distinct teams from alive without replacement.
        - Note: If we choose with replacement, the same team can be picked twice.
        '''
        alive = np.where(n > 0)[0]
        a, b = rng.choice(alive, size=2, replace=False)
        ### Assigning team_i (team with smaller index) and team_j (team with higher index)
        i_k, j_k = min(a, b), max(a, b)

        ## Storing the match information.
        record = { "Event": k,
                   "Team i": i_k + 1,
                   "Team j": j_k + 1,
                   **{f"n_{t+1}": n[t] for t in range(N)}
                }

        ## Weight probability
        prob_vec = np.zeros_like(n, dtype=float)
        prob_vec[alive] = np.exp( gamma*np.log(n[alive]) + alpha[alive])

        p_a_win = (prob_vec[a]) / (prob_vec[a] + prob_vec[b])

        winner = a if rng.random() < p_a_win else b
        loser = b if winner == a else a

        record["Elimination"] = loser + 1

        match_records.append(record)
        
        n[loser] -= 1
        k += 1

    Pw_match_data = pd.DataFrame(match_records)

    return Pw_match_data

###### =============================================================
###### -------- Simulating elimination match data ------------------
###### =============================================================
def simulate_elimination_competition(alpha, n0, data_rand_state=None):
    """ 
    Simulate match data from the elimination model

    Inputs
    -----------
    alpha           : ndarray of team strengths
    n0              : int (initial team size, Solo =1, Duo = 2, Squad =4)
    data_rand_state : Set to None to to generate different data on each run

    Output 
    ---------
    Elim_match_data : Pandad DataFrame of sequence of team sizes 

    Refer to the "STATISTICAL MODEL FOR MULTI-PLAYER GAME DYNAMICS" for details of the model
    """
    
    if data_rand_state is not None:
        data_rand_state = int(data_rand_state)

    ## setting random_state = None generate different data anytime the code is run
    rng = np.random.default_rng(data_rand_state)

    n0 = int(n0)                      ## Initial team sizes
    alpha = to_numpy(alpha).ravel()   ## an array of latent team strength

    beta = -1.25                      ## team size sensitivity parameter estimated from real match data

    N = len(alpha)                    ## The length of alpha (number of teams)
    n = np.full(N, n0, dtype=int)     ## Initialize team size vector

    match_records = []                ## Initialize matrix to store match data 
    
    k = 1

    while np.sum(n > 0)>1:

        alive = np.where(n > 0)[0]

        weights= np.zeros_like(n, dtype=float)

        weights[alive] = np.exp( beta*np.log(n[alive]) - alpha[alive])

        p = weights / np.sum(weights)

        loser_ind = rng.multinomial(1, p)

        loser_team = np.arange(N)[np.argmax(loser_ind)]

        other_candidates = alive[alive != loser_team]
        other_team = rng.choice(other_candidates)

        i_k, j_k = min(loser_team, other_team), max(loser_team, other_team)

        record = { "Event": k,
                   "Team i": i_k + 1,
                   "Team j": j_k + 1,
                   **{f"n_{t+1}": n[t] for t in range(N)},
                   "Elimination": loser_team + 1
                }
        
        match_records.append(record)

        n[loser_team] -= 1

        k += 1

    Elim_match_data = pd.DataFrame(match_records)
    
    return Elim_match_data


###### =============================================================
###### ------ Stacking k-matches into one dataset ------------------
###### =============================================================
def combined_match_data(alpha, n0, num_of_stack_data, elimination_data=True, data_rand_state=None):
    """
    Stack multiple simulated competitions into one dataset.
    
    Parameters
    ----------
    alpha             : array-like (latent team strengths)
    n0                : int (Initial team size)
    num_of_stack_data : int (Number of competitions/matches to be stacked together )
    data_rand_state   : int, (optional, Random seed for reproducibility)

    elimination_data  : True/False generate elimination/pairwise data respectively

    Returns
    -------
    pd.DataFrame for a Combined dataset of all stacked competitions
    """

    alpha = to_numpy(alpha).ravel()  
    
    n0 = int(n0) 
    num_of_stack_data = int(num_of_stack_data)

    if data_rand_state is not None:
        data_rand_state = int(data_rand_state)

    all_games = []

    for m in range(num_of_stack_data):
        rs = None if data_rand_state is None else data_rand_state + m

        if elimination_data is True:
            df = simulate_elimination_competition(alpha,  n0, data_rand_state=rs)
        else:
            df = simulate_pairwise_competition(alpha, n0, data_rand_state=rs)
            
        all_games.append(df)
    
    combine_df = pd.concat(all_games, axis=0, ignore_index=True)

    return combine_df

###### =============================================================
###### ------ Model free Survival based ranking --------------------
###### =============================================================
def ranking_based_on_team_size_per_encounter(NumOfTeams, matN):
    """ 
    - The function rank the teams based on average team size per encounter (model free)
    - The model compute average players per team for a given match.
    - Its uses the average team sizes as a measure of strength to rank the teams.
    - We assume here that larger teams are stronger than the smaller teams.

    Inputs
    -----------
    NumOfTeams  : int (Number of teams in the tournament )
    matN        : T by N matrix of team sizes in the tournament

    Output 
    ----------
    N by 3 matrix consisting of team index, position and survival score (average team size per encounter)
    """

    NumOfTeams = int(NumOfTeams)       ## number of physical teams in the game
    matN = to_numpy(matN)              ## the size matrix

    _, N = matN.shape                  ## N is columns of team size matrix 

    if NumOfTeams < N:                 ## Checking if the game setting was involved or not
        matN = matN[:, :-1]            ## Excluding the game setting if it was involved 
        N -= 1

    Avg_team_size = np.mean(matN, axis=0) ## Average team size per encounter
    
    team_ids = np.arange(1, N + 1)           ## Creating the team IDs
    
    surv_pos = rankdata(-Avg_team_size, method='average')
    
    team_size_ranks = np.column_stack((team_ids, surv_pos, np.round(Avg_team_size,5)))


    return team_size_ranks

###### =============================================================
###### ------ Strength based ranking (Model dependent) -------------
###### =============================================================
def ranking_by_team_strength(NumOfTeams, alpha):
    """ 
    - The function rank the teams based on latent strength alpha (true or estimated)
    - Larger alpha => stronger team.
    - The ranking exclude the game setting if it get involved in the elimination. 

    Inputs
    -----------
    NumOfTeams  : int (Number of teams in the tournament )
    alpha       : ndarray of team strengths

    Output 
    ----------
    N by 3 matrix consisting of team index, position and team strengths (alpha)
    """

    NumOfTeams = int(NumOfTeams)     ## number of physical teams in the game
    
    alpha = to_numpy(alpha).ravel()  ## Team strengths

    N = len(alpha)

    if NumOfTeams < N:               ## Checking to ensure game setting is excluded
        alpha = alpha[:-1]
        N -= 1

    TeamID = np.arange(1, N + 1)     ## creating team IDs for ranking

    alpha_pos = rankdata(-alpha, method='average')

    strength_rank = np.column_stack((TeamID, alpha_pos, np.round(alpha,5)))

    return strength_rank

######### ======================================================================
def combined_ranking(NumOfTeams, alpha, matN):
    '''
    - This function rank the teams based on the combination of average team size and latent strengths
    - It average the survival (model free) position and the strength (model based) position
    - By default, we set the weight to 0.5
   
    Inputs
    --------------
    NumOfTeams  : int (Number of teams in the tournament )
    alpha       : ndarray of team strengths
    matN        : T by N matrix of team sizes in the tournament

    Output
    --------------
    N by 3 matrix consisting of team index, position and average position score
    '''

    NumOfTeams = int(NumOfTeams)  
    size_rank = ranking_based_on_team_size_per_encounter(NumOfTeams, matN)
    strength_rank  = ranking_by_team_strength(NumOfTeams, alpha)

    size_pos = size_rank[:,1]
    strength_pos = strength_rank[:,1]
    
    combined_score = 0.5*(size_pos + strength_pos)

    TeamID = np.arange(1, NumOfTeams + 1)     ## creating team IDs for ranking

    comb_pos = rankdata(combined_score, method='average')

    combine_ranking = np.column_stack( ( TeamID, comb_pos, np.round(combined_score,5) ) )

    return combine_ranking

###### =============================================================
###### ------ Definining the model ---------------------------------
###### =============================================================
def build_B_sum_to_zero(N):
    """ 
    The function builds the identifiability matrix B = N x (N -1) such:
    1. alpha = B @ mu
    2. sum(alpha)= 0

    Construction
    -------------------
    - identity for first N-1 rows
    - last row = -1

    - Note that N (number of teams) must be at least 2 
    """
    N = int(N) ## Ensuring N is an integer

    if N < 2:
        raise ValueError("There has to be at least 2 teams")

    B = np.zeros((N, N - 1))
    B[:N - 1, :N - 1] = np.eye(N - 1)
    B[N - 1, :] = -1.0

    return B

####### ==================================================================
####### ------------ Survival-Based Initial Values -----------------------
####### ==================================================================
def compute_initial_mu(matN):
    """
    In fitting the model, we used the average team size from the data as the initial condition.
    The function computes the survival based initial condition.
    This provides a data-driven initialization reflecting team strength
    
    Parameters
    ----------
    matN : Match data with team size columns (n_1, n_2, ..., n_N)
    
    Returns
    -------
    np.ndarray
        Initial mu values (N-1 dimensional) based on survival rankings
    """    
    N = matN.shape[1]  # Number of teams
    
    # Compute average team size for each team (survival proxy)
    avg_team_sizes = matN.mean(axis=0)

    centered_team_size = avg_team_sizes - np.mean(avg_team_sizes)  ## recenter the team sizes
    
    ### Recall that alpha = B*mu, so here we solve for mu given alpha and B
    B = build_B_sum_to_zero(N)
    
    mu_init = np.linalg.lstsq(B, centered_team_size, rcond=None)[0]
    
    return mu_init

###### =============================================================
###### ----------------- Elimination matrix Ey ---------------------
###### =============================================================
def build_elimination_matrix_Ey(N, y):
    """
    - Each row of the elimination matrix captures the team that lost a player at that very encounter
    - Elimination matrix Ey = T x N corresponding to the observed order of elimination.
    - Each row is a canonical basis with 1 at the entry corresponding to the index of 
      the team that lost player
    """
    N = int(N)
    ## Ensure y is properly converted and flattened
    y = to_numpy(y).flatten()

    T = len(y)
    ## Explicit conversion to int for indexing
    cols = np.asarray(y, dtype=int) - 1

    Ey = np.zeros((T, N), dtype=np.uint8)
    Ey[np.arange(T), cols] = 1

    return Ey

###### =====================================================================
###### ---------- Team participating matrix Ei & Ej ------------------------
###### =====================================================================
def Build_team_participating_matrices(matN, team_i, team_j):
    ''' 
    - Team participation matrices E_i and E_j corresponding to team i and team j involved 
      in the various ecounters. 
    - Each row of E_i and E_j are zero except the entry corresponding to team_i and team_j 
      involved in the t^th encounter.
    - Refer to "STATISTICAL MODEL FOR MULTI-PLAYER GAME DYNAMICS" for more details
    '''

    matN = to_numpy(matN)
    T, N = matN.shape

    ## Ensure team indices are properly converted, flattened and indexed from zero
    team_i = to_numpy(team_i).flatten().astype(int) - 1
    team_j = to_numpy(team_j).flatten().astype(int) - 1


    Ei = np.zeros((T, N), dtype=np.uint8)
    Ej = np.zeros((T, N), dtype=np.uint8)

    Ei[np.arange(T), team_i] = 1
    Ej[np.arange(T), team_j] = 1

    return Ei, Ej

#### ===================================================================
#### ------- Computing the log and exponent of team sizes --------------
#### ===================================================================
def log_and_beta_exponent_of_team_sizes(beta, matN):
    '''
    The function computes:
    1. n^{beta} = n_{t,i}^{beta} if n_{t,i} != 0 and 
    2. log(n) = ln(n_{t,i}) if n_{t,i} != 0 

    '''
    matN = to_numpy(matN)

    alive = matN > 0

    n_exp_beta = np.zeros_like(matN, dtype=float)
    n_exp_beta[alive] = matN[alive]**beta

    LogN = np.zeros_like(matN, dtype=float)
    LogN[alive] = np.log(matN[alive])

    log_matrices = {"n_exp_beta": n_exp_beta,
                    "LogN": LogN
                    }

    return log_matrices

#### =============================================================
#### ---------- Elimination likelihood matrices ------------------
#### =============================================================
def elim_likelihood_matrices(theta, matN, y):
    """
    The function:
    - Builds various matrices (diag(My), Ny, P) needed to construct the likelihood estimate
      and the gradient for the elimination model
    Recall that: 
    - n_exp_beta_i = n_{t-1, i}^beta
    - LogN_i = log(n_{t-1, i}) if n_{t-1,i} > 0 and zero when n_{t-1,i} = 0

    For more details, see appendix of  "STATISTICAL MODEL FOR MULTI-PLAYER GAME DYNAMICS"
    """

    matN = to_numpy(matN)
    y = to_numpy(y).ravel()
    theta = to_numpy(theta).ravel()
    beta = theta[0]
    # gamma = theta[1]       ## This is not used here
    mu = theta[2:].reshape(-1,1)

    _, N = matN.shape
    B = build_B_sum_to_zero(N)
 
    log_matrices = log_and_beta_exponent_of_team_sizes(beta, matN)
    n_exp_beta = log_matrices["n_exp_beta"]
    LogN = log_matrices["LogN"]

    ###------- New formulation--------
    alpha = B @ mu
    alpha_clipped = np.clip(alpha, -50, 50)    ## clip alpha to ensure stability
    V = np.exp(-alpha_clipped)

    Ey = build_elimination_matrix_Ey(N, y)

    diagMy = np.diag(LogN@Ey.T).reshape(-1,1)

    a_beta = (beta*diagMy - Ey@alpha).reshape(-1,1)

    eps = 1e-12
    Ny = (n_exp_beta@V).reshape(-1,1)
    Ny = np.maximum(Ny, eps)

    P_elim = (n_exp_beta*V.T)/Ny

    elim_matrices = { 'a_beta':a_beta,
                      'Ny':Ny, 
                      'P_elim':P_elim }
    
    return elim_matrices

#### ===========================================================
#### ---------- Pairwise likelihood matrices -------------------
#### ===========================================================
def Pairwise_likelihood_matrices(theta, matN, y, team_i, team_j):
    """
    - The function computes the the matrices and vectors needed to formulate 
      the pairwise nodel in compact form.
    - For detailed explanation, see the  appendix of "STATISTICAL MODEL FOR MULTI-PLAYER GAME DYNAMICS"
    """
    matN = to_numpy(matN)
    team_i = to_numpy(team_i)
    team_j = to_numpy(team_j)
    y = to_numpy(y)

    theta = to_numpy(theta).ravel()
    # beta = theta[0]            ## Note needed here
    gamma = theta[1]
    mu = theta[2:].reshape(-1,1)

    _, N = matN.shape
    B = build_B_sum_to_zero(N)

    alpha = B @ mu

    ## Team participating matrix
    Ei, Ej = Build_team_participating_matrices(matN, team_i, team_j)

    ##  Observed elimination for team i and team j
    Si = np.asarray((team_i == y), dtype=int).reshape(-1,1)
    Sj = np.asarray((team_j == y), dtype=int).reshape(-1,1)
    
    ## Team sizes for team i and team j
    ni = np.diag(matN @ Ei.T).reshape(-1, 1)
    nj = np.diag(matN @ Ej.T).reshape(-1, 1)

    ## Log of team sizes for team i and team j
    Li = np.log(ni).reshape(-1, 1)
    Lj = np.log(nj).reshape(-1, 1)
    
    ## Linear prodictors for team i and team j
    ai_gamma = gamma*Li + Ei @ alpha
    aj_gamma = gamma*Lj + Ej @ alpha

    ## win probabilities and their denominators 
    LogExp_ai_aj = np.logaddexp(ai_gamma, aj_gamma)

    P_pw = expit(ai_gamma - aj_gamma)

    pw_matrices = { 'ai_gamma': ai_gamma,
                    'aj_gamma': aj_gamma,
                    'Si': Si,
                    'Sj': Sj,
                    'Li': Li,
                    'Lj': Lj,
                    'LogExp_ai_aj': LogExp_ai_aj,
                    'P_pw': P_pw  
                    }
    
    return pw_matrices

##### ============================================================
##### ---------- combined likelihood estimate --------------------
##### ============================================================
def comb_nll_and_grad(theta, lam, matN, y, team_i, team_j, omega):
    """ 
    - This function defines the likelihood estimate and its gradient for the combined model.
    - Based on the choice of omega, we can run just the elimination, pairwise or the combined model.
    - For detail construction of this refer the  appendix of "STATISTICAL MODEL FOR MULTI-PLAYER GAME DYNAMICS"

    Inputs
    ----------------
    theta    : parameter values ( theta = [beta, gamma, mu])
    lam      : Regularization/penalty parameter 
    matN     : Team size matrix
    y        : vector of order of elimination
    team_i   : first team for each encounter 
    team_j   : second team for each encounter 
    omega    : weight assigned to elimination and pairwise model 

    Recall that: 
    1. elim_matrices = elimination_likelihood_matrices(theta, matN, y)

       elim_matrices = {'a_beta':a_beta,
                        'Ny':Ny, 
                        'P_elim':P_elim
                        }

    2. pw_matrices = Pairwise_likelihood_matrices(theta, matN, y, team_i, team_j)

        pw_matrices = { 'ai_gamma': ai_gamma,
                        'aj_gamma': aj_gamma,
                        'Si': Si,
                        'Sj': Sj,
                        'Li': Li,
                        'Lj': Lj,
                        'LogExp_ai_aj': LogExp_ai_aj,
                        'P_pw': P_pw
                        }
    """

    omega = np.asarray(omega, dtype=float).ravel()
    if omega.size != 2:
        raise ValueError("omega must have length 2")

    theta = to_numpy(theta).ravel()
    beta = theta[0]
    # gamma = theta[1]
    mu = theta[2:].reshape(-1,1)

    lam = float(lam)
    
    matN = to_numpy(matN)
    y = to_numpy(y)

    T, N = matN.shape
    OneT = np.ones((T, 1))
    OneN = np.ones((N, 1))
    B = build_B_sum_to_zero(N)

    alpha = B @ mu

    Ey = build_elimination_matrix_Ey(N, y)

    Ei, Ej = Build_team_participating_matrices(matN, team_i, team_j)

    log_matrices = log_and_beta_exponent_of_team_sizes(beta, matN)

    LogN = log_matrices["LogN"]


    ## Extrating the matrices and vectors for the elimination model
    elim_matrices = elim_likelihood_matrices(theta, matN, y)
    Ny = to_numpy(elim_matrices['Ny']).reshape(-1,1)
    a_beta = to_numpy(elim_matrices['a_beta']).reshape(-1,1)
    P_elim = to_numpy(elim_matrices['P_elim'])

    ## Extrating the matrices and vectors for the pairwise model
    pw_matrices = Pairwise_likelihood_matrices(theta, matN, y, team_i, team_j)
    ai_gamma = to_numpy(pw_matrices['ai_gamma']).reshape(-1,1)
    aj_gamma = to_numpy(pw_matrices['aj_gamma']).reshape(-1,1)
    Si = to_numpy(pw_matrices['Si']).reshape(-1,1)
    Sj = to_numpy(pw_matrices['Sj']).reshape(-1,1)
    Li = to_numpy(pw_matrices['Li']).reshape(-1,1)
    Lj = to_numpy(pw_matrices['Lj']).reshape(-1,1)
    LogExp_ai_aj = to_numpy(pw_matrices['LogExp_ai_aj']).reshape(-1,1)
    P_pw = to_numpy(pw_matrices['P_pw']).reshape(-1,1)

    ## Defining the combined likelihood estimate (not negative log-likelihood)
    elim_lik = omega[0] * (OneT.T @ a_beta - OneT.T @ np.log(Ny)) - (lam / 2) * (alpha.T @ alpha)
    pw_lik = omega[1] * (Sj.T @ ai_gamma + Si.T @ aj_gamma - OneT.T @ LogExp_ai_aj) - (lam / 2) * (alpha.T @ alpha)

    comb_lik = elim_lik + pw_lik


    ## Defining the gradient of the likelihood estimate
    el_grad_beta =omega[0]*( OneT.T@( (np.diag(LogN @ Ey.T)).reshape(-1,1) - (P_elim*LogN)@OneN ) )
    el_grad_alpha = (P_elim.T @ OneT) - (Ey.T @ OneT)
    el_grad_mu = omega[0]*(B.T @ el_grad_alpha) - lam * (B.T @ alpha)
    

    elim_grad = np.zeros(len(theta), dtype=float)
    elim_grad[0] = el_grad_beta.item()
    elim_grad[1] = 0.0
    elim_grad[2:] = el_grad_mu.ravel()

    ## Finding the gradient of the pairwise model
    pw_grad_gamma = omega[1]*( Li.T@( Sj - P_pw ) + Lj.T@ (Si - (1 - P_pw) ) )
    pw_grad_alpha = Ei.T@(Sj - P_pw) + Ej.T@(Si - (1 - P_pw) )
    pw_grad_mu = omega[1]*(B.T @ pw_grad_alpha) - lam * (B.T @ alpha)

    pw_grad = np.zeros(len(theta), dtype=float)
    pw_grad[0] = 0.0
    pw_grad[1] = pw_grad_gamma.item()
    pw_grad[2:] =  pw_grad_mu.ravel()

    comb_grad = elim_grad + pw_grad

    return comb_lik, comb_grad 

### ================================================================
### ------ Extracting just the likelihood estimate -----------------
### ================================================================
def comb_nll_only(theta, lam, matN, y, team_i, team_j, omega):
    """
    - Defining only the negative log-likelihood estimate for the combined model
    """
    like_est, _ = comb_nll_and_grad(theta, lam, matN, y, team_i, team_j, omega)

    comb_nll = (-like_est).item()

    return comb_nll

### ================================================================
### -- Extracting just the gradient of the likelihood estimate -----
### ================================================================
def comb_grad_nll_only(theta, lam, matN, y, team_i, team_j, omega):
    """
    - Defining just the gradient of the negative log-likelihood estimate for the combined model
    """

    theta = np.asarray(theta, dtype=float).ravel().reshape(-1, 1)

    _, grad_alpha = comb_nll_and_grad(theta, lam, matN, y, team_i, team_j, omega)

    gradient_mu = (-grad_alpha).ravel()

    return gradient_mu 

####### =============================================================
####### --- Optimization: Fiting the logistic regression model ------
####### =============================================================
def comb_fit_logistic_model(df, lam, omega, NumOfTeams, seedNum, alpha_max, mu0=None, real_data = False):
    """ 
    - Fitting the logistic regression model to eatimate the optimal team strength 
      and the team size sensitivity parameter beta and gamma.
    - Recall that combine_model = omega_1*elimination_model + omega_2*pairwise_model
    - Depending on the choice of omega = (omega_1, omega_2), we can simulate just the elimination, 
      pairwise or the combined model 

    Data Selection:
    ----------------------
    real_data = False ==> using simulated data
    real_data = True ==> using real match data

    Inputs
    ----------------------
    df          : match data 
    lam         : constrain penalty parameter
    omega       : weight assigned to the different models
    NumOfTeams  : Number of teams involved in the tournament 
    seedNum     : random seed for reproducibility
    mu0         : Initial parameter values 
    real_data   : Type of data used (simulated or actual match)
    alpha_max   : Maximum strength assigned to any team (-alpha_max <= alpha <= alpha_max)
    """

    NumOfTeams = int(NumOfTeams)
    seedNum = int(seedNum)
    lam = float(lam)
    alpha_max = float(alpha_max)

    ## Ensure omega is proper numpy array
    omega = np.asarray(omega, dtype=float).ravel()

    if omega.size != 2:
        raise ValueError("omega must have length 2")
    
    ## *******************************************************************************
    ## -- Extracting team size matrix, team indices, and observed order or elimination
    if real_data is False:
        ## Extracting information from simulated data
        n_cols = [col for col in df.columns if str(col).startswith("n_")]
        matN = to_numpy(df[n_cols])
        team_i = df["Team i"].astype(int).to_numpy()
        team_j = df["Team j"].astype(int).to_numpy()
        y = df["Elimination"].astype(int).to_numpy()
    
    else:
        ## Extracting information from real match data
        matN = df.iloc[:,11:-2]
        team_i = df["Team i (ID)"].astype(int).to_numpy()  
        team_j = df["Team j (ID)"].astype(int).to_numpy()
        y = df["Elimination"].astype(int).to_numpy()

    if mu0 is None:
        ## when using survival based initialization for optimization
        mu0 = compute_initial_mu(matN)
        mu0 = to_numpy(mu0).ravel()
    else:
        ## when using other initial parameter set for optimization
        mu0 = np.asarray(mu0, dtype=float).ravel()

    theta0 = np.concatenate(([1.0, 1.0], mu0))

    ## -------- Fitting the model ------------------------------------------
    comb_res = minimize( fun=comb_nll_only, x0=theta0,  jac=comb_grad_nll_only,
                         args=(lam, matN, y, team_i, team_j, omega), method="BFGS",
                         options={"gtol": 1e-8, "maxiter": 250, "disp": False})
    
    theta_hat = comb_res.x
    beta_hat = theta_hat[0]
    gamma_hat = theta_hat[1]
    mu_hat = to_numpy(theta_hat[2:]).reshape(-1,1)

    _, N = matN.shape
    B = build_B_sum_to_zero(N)

    alpha_hat = to_numpy(B @ mu_hat).ravel()
    alpha_hat = np.round(alpha_hat, 5)

    opt_lik_est = np.round(comb_res.fun, 5)

    #### **************************************************
    TrueAlpha = generate_alpha(NumOfTeams, alpha_max, seedNum)

    #### **********************************************************************
    ''' Here, we computes the ranking based on the true team strength ''' 
    TrueRank = ranking_by_team_strength(NumOfTeams, TrueAlpha)

    ''' Here, we computes the ranking based on the predicted team strength '''
    PredRank = ranking_by_team_strength(NumOfTeams, alpha_hat)

    ''' Here, we computes the ranking based on the averaage team size per encounter '''
    TS_rank = ranking_based_on_team_size_per_encounter(NumOfTeams, matN)
    
    """Here, we compute ranking based on the combination of the two ranking system """
    CombRank = combined_ranking(NumOfTeams, alpha_hat, matN)


    #### **********************************************************************
    #### --- Model performance in recovering team strengths -------------------
    if NumOfTeams < len(alpha_hat):        ## Ensure game setting is exluded from ranking
        alpha_hat = alpha_hat[:-1]

    Alpha_diff = (TrueAlpha - alpha_hat).reshape(-1,1)

    alpha_rmse = np.sqrt(np.sum(Alpha_diff**2) / (NumOfTeams))  # Root mean square error
    alpha_rmse = np.round(alpha_rmse, 5)

    alpha_mae = np.mean(np.abs(Alpha_diff))                     # Mean absolute error
    alpha_mae = np.round(alpha_mae, 5)


    #### *******************************************************************
    #### --- Model performance in recovering rankings ----------------------
    Pred_rank_diff = (TrueRank[:,1] - PredRank[:,1]).reshape(-1,1)   # Rank difference
    pred_rmse = np.sqrt(np.sum(Pred_rank_diff**2) / (NumOfTeams))    # Root mean square error
    pred_rmse = np.round(pred_rmse, 5)

    pred_mae = np.mean(np.abs(Pred_rank_diff))                      # Mean absolute error
    pred_mae = np.round(pred_mae, 5)

    pred_comp = (TrueRank[:,1] == PredRank[:,1]).astype(int)        # compare predictions
    pred_correct = np.sum(pred_comp)/len(pred_comp)                 # Fraction of correct predictions
    pred_correct = np.round(pred_correct, 5)
    pred_incorrect = 1 - pred_correct                               # Fraction of incorrect predictions

    pred_SpearmanR, _ = spearmanr(TrueRank[:,1], PredRank[:,1])     # Spearmans rank correlation
    pred_SpearmanR = np.round(pred_SpearmanR, 5)

    pred_Ranking = pd.DataFrame({ 'Team ID': PredRank[:,0],
                                  'pred_pos': PredRank[:,1],
                                  'Pred_alpha': PredRank[:,2],
                                  'True_Pos': TrueRank[:,1],
                                  'True_alpha': TrueRank[:,2],
                                })
    
    #### *******************************************************************
    #### --- Comparing survival ranking to the true ranking ----------------
    TS_rank_diff = (TrueRank[:,1] - TS_rank[:,1]).reshape(-1,1)   # Rank difference
    TS_rmse = np.sqrt(np.sum(TS_rank_diff**2) / (NumOfTeams))     # Root mean square error
    TS_rmse = np.round(TS_rmse, 5)

    TS_mae = np.mean(np.abs(TS_rank_diff))                        # Mean absolute error
    TS_mae = np.round(TS_mae, 5)

    TS_comp = (TrueRank[:,1] == TS_rank[:,1]).astype(int)         # compare predictions
    TS_correct = np.sum(TS_comp)/len(TS_comp)                     # Fraction of correct predictions
    TS_correct = np.round(TS_correct, 5)
    TS_incorrect = 1 - TS_correct                                # Fraction of incorrect predictions

    TS_SpearmanR, _ = spearmanr(TrueRank[:,1], TS_rank[:,1])      # Spearmans rank correlation
    TS_SpearmanR = np.round(TS_SpearmanR, 5)

    TS_Ranking = pd.DataFrame({ 'Team_ID': TS_rank[:,0],
                                'Avg_pos':TS_rank[:,1],
                                'Avg_team_size': TS_rank[:,2],
                                'True_pos': TrueRank[:,1],
                                'True_alpha': TrueRank[:,2]
                                })
    
    #### *******************************************************************
    #### --- Using combine ranking method to access model performance ------
    comb_rank_diff = (TrueRank[:,1] - CombRank[:,1]).reshape(-1,1)   # Rank difference
    comb_rmse = np.sqrt(np.sum(comb_rank_diff**2) / (NumOfTeams))                # Root mean square error
    comb_rmse = np.round(comb_rmse, 4)

    comb_mae = np.mean(np.abs(comb_rank_diff))                     # Mean absolute error
    comb_mae = np.round(comb_mae, 4)

    comb_comp = (TrueRank[:,1] == CombRank[:,1]).astype(int)         # compare predictions
    comb_correct = np.sum(comb_comp)/len(comb_comp)                # Fraction of correct predictions
    comb_correct = np.round(comb_correct, 4)
    comb_incorrect = 1 - comb_correct                    # Fraction of incorrect predictions

    comb_SpearmanR, _ = spearmanr(TrueRank[:,1], CombRank[:,1])      # Spearmans rank correlation
    comb_SpearmanR = np.round(comb_SpearmanR, 4)

    Comb_Ranking = pd.DataFrame({ 'Team_ID': CombRank[:,0],
                                  'Comb_pos': CombRank[:,1],
                                  'Comb_score': CombRank[:,2],
                                  'True_pos': TrueRank[:,1],
                                  'True_alpha': TrueRank[:,2]
                                })
 
    ### ------------------------------------------
    Joint_Table = pd.DataFrame({ 'Team_ID': TrueRank[:,0],
                                 'True_Pos': TrueRank[:,1],
                                 'True_alpha': TrueRank[:,2],
                                 'Pred_Pos': PredRank[:,1],
                                 'Pred_alpha': PredRank[:,2],
                                 'Team_Pos': TS_rank[:,1],
                                 'Avg_team_size': TS_rank[:,2],
                                 'Comb_Pos': CombRank[:,1],
                                 'Comb_score': CombRank[:,2]
                                })
   
    #### **************************************************
    result = {  ## -- some simulated results --
                'Team_ids': np.arange(1, len(alpha_hat) + 1),
                'beta_hat': beta_hat,
                'gamma_hat': gamma_hat,
                'alpha_hat': alpha_hat,
                'identifiability_cond': np.sum(alpha_hat),
                'mu': mu_hat,
                'Opt_lik_est': opt_lik_est,
                "Joint_Table": Joint_Table, 
                'success': bool(comb_res.success),

                ## -- Comparison between true and predicted alpha--
                'RMSE between alpha': alpha_rmse,
                'MAE between alpha': alpha_mae,

                ## -- Ranking based on predicted team strength --
                'Predicted Ranking': pred_Ranking,
                'Predicted RMSE': pred_rmse,
                'Predicted MAE': pred_mae,
                'Predicted Spearmans': pred_SpearmanR,
                'Perfect prediction':  pred_correct,
                'Wrong prediction': pred_incorrect,

                ## -- Ranking based on predicted team strength --
                'Team size Ranking': TS_Ranking,
                'Team size RMSE': TS_rmse,
                'Team size MAE': TS_mae,
                'Team size Spearmans': TS_SpearmanR,
                'Team size perfect_pred':  TS_correct,
                'Team size wrong_pred': TS_incorrect, 
                
                ## -- Ranking based on combined ranking methods --
                'Combined Ranking': Comb_Ranking,
                'Combined RMSE': comb_rmse,
                'Combined MAE': comb_mae,
                'Combined Spearmans': comb_SpearmanR,
                'Combined perfect_pred':  comb_correct,
                'Combined worng_pred': comb_incorrect
                }

    return result   


####### =============================================================
####### ------ Generating folds to perform a cross validation -------
####### =============================================================
def generate_cv_folds(df, n_splits, seedNum = None):
    """
    ----- Generate Cross-Validation Folds ------------------

    Generate cross-validation fold assignments for a dataset.
    
    This allows you to inspect and test fold composition before running CV.
    
    Parameters
    ------------------------------------
    df         : pd.DataFrame  ( Match data)
    n_splits   : int, optional ( Number of folds (default: 6))
    seedNum    : int, optional ( Random seed for reproducibility (default: 35))

    Returns
    -------------------------------------
    dict
        'fold_indices': List of (train_idx, test_idx) tuples for each fold
        'fold_labels': Array of fold assignments (0 to n_splits-1) for each row
        'n_splits': Number of splits
        'n_observations': Total number of observations
        'fold_sizes': List of test set sizes for each fold
    """
    
    n_splits = int(n_splits)

    if seedNum is not None:
       seedNum = int(seedNum)
    
    n_obs = len(df)      ## Identifier for data set used for testing for each fold
    
    # Generate fold assignments
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seedNum)
    
    fold_indices = []       ## A tuple recording train_indice and test_indices for each fold
    fold_labels = np.zeros(n_obs, dtype=int)  
    fold_sizes = []         ## Size of testing data for each fold
    
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(df)):
        fold_indices.append((train_idx.tolist(), test_idx.tolist()))
        fold_labels[test_idx] = fold_num      ## list of indices selected for testing in each fold
        fold_sizes.append(len(test_idx))
    
    return { 'fold_indices': fold_indices,
             'fold_labels': fold_labels.tolist(),
             'n_splits': n_splits,
             'n_observations': n_obs,
             'fold_sizes': fold_sizes }


######## ============================================================
######## ------ Performing cross validation in Parallel  ------------
######## ============================================================
def fit_logistic_cv_model(df, lambdas, omega, n_splits, seedNum=None, mu0=None, 
                         fold_assignments=None, use_hot_start=True, real_data=False):
    """
    For faster implementation, this function performs CV in parallel 
    Perform k-fold cross-validation to find the optimal regularization parameter.
    
    Parameters
    ----------------------
    df       : pd.DataFrame (Match data with team sizes and elimination outcomes)
    lambdas  : array-like  (Sequence of lambda values to test)
    omega    : array-like of length 2 ( Weights [elimination_weight, pairwise_weight])
    n_splits : int, optional ( Number of folds for cross-validation (default: 6))
               Ignored if fold_assignments is provided
    seedNum  : int, optional ( Random seed for reproducibility (default: 35))
               Ignored if fold_assignments is provided
    mu0      : array-like, optional
               Initial parameter values (default: survival-based for lambda=0)
               If None, uses survival rankings for lambda=0, then hot start for others
    fold_assignments : dict, optional
                Pre-generated fold assignments from generate_cv_folds()
                If provided, uses these instead of generating new folds
    use_hot_start : bool, optional
                If True, use solution from previous lambda as initialization (default: True)
                Requires lambdas to be sorted (smallest to largest recommended)
    
    Returns
    -----------------------
    dict
        'Results': (n_lambdas, 2) array of [lambda, avg_nll]
        'OptimalVal': [optimal_lambda, min_avg_nll]
        'Opt_lambda': optimal lambda value
        'CV_Details': list of nll values for each fold and lambda
        'fold_assignments': The fold assignments used (for inspection)
    """

    lambdas = np.asarray(lambdas, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()

    if omega.size != 2:
        raise ValueError("omega must have length 2")

    if use_hot_start:
        lambda_sort_idx = np.argsort(lambdas)
        lambdas_sorted = lambdas[lambda_sort_idx]
    else:
        lambdas_sorted = lambdas
        lambda_sort_idx = np.arange(len(lambdas))

    # ---- Extract matrix ----
    if not real_data:
        n_cols = [col for col in df.columns if str(col).startswith("n_")]
        matN = to_numpy(df[n_cols])
    else:
        matN = to_numpy(df.iloc[:, 11:-2])

    _, N = matN.shape

    # ---- Initialization ----
    if mu0 is None:
        mu0 = compute_initial_mu(matN)
        mu0 = to_numpy(mu0).ravel()
    else:
        mu0 = np.asarray(mu0, dtype=float).ravel()

    if len(mu0) != N - 1:
        raise ValueError("mu0 must be length N-1")

    theta0 = np.concatenate(([1.0, 1.0], mu0)).flatten()

    # ---- Fold assignments ----
    if fold_assignments is not None:
        fold_indices = fold_assignments['fold_indices']
        fold_info = fold_assignments
    else:
        fold_info = generate_cv_folds(df, n_splits=n_splits, seedNum=seedNum)
        fold_indices = fold_info['fold_indices']

    lambda0 = 0.0
    avg_nll = []

    fold_theta_inits = [theta0.copy() for _ in range(len(fold_indices))]

    # ---- Parallel worker ----
    def process_fold(fold_num, train_idx, test_idx, lam, current_init):

        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

        if not real_data:
            n_cols = [col for col in df_train.columns if str(col).startswith("n_")]
            train_matN = to_numpy(df_train[n_cols])
            train_team_i = to_numpy(df_train["Team i"])
            train_team_j = to_numpy(df_train["Team j"])
            train_y = to_numpy(df_train["Elimination"])

            n_cols = [col for col in df_test.columns if str(col).startswith("n_")]
            test_matN = to_numpy(df_test[n_cols])
            test_team_i = to_numpy(df_test["Team i"])
            test_team_j = to_numpy(df_test["Team j"])
            test_y = to_numpy(df_test["Elimination"])
        else:
            train_matN = to_numpy(df_train.iloc[:, 11:-2])
            train_team_i = to_numpy(df_train["Team i (ID)"])
            train_team_j = to_numpy(df_train["Team j (ID)"])
            train_y = to_numpy(df_train["Elimination"])

            test_matN = to_numpy(df_test.iloc[:, 11:-2])
            test_team_i = to_numpy(df_test["Team i (ID)"])
            test_team_j = to_numpy(df_test["Team j (ID)"])
            test_y = to_numpy(df_test["Elimination"])

        current_init = current_init.flatten()

        train_res = minimize(fun=comb_nll_only, x0=current_init, jac=comb_grad_nll_only,
                             args=(lam, train_matN, train_y, train_team_i, train_team_j, omega),
                             method='BFGS', options={"gtol": 1e-8, "maxiter": 250, "disp": False}
                             )

        train_theta_hat = train_res.x

        test_cost_nll = comb_nll_only(train_theta_hat, lambda0, test_matN, test_y, 
                                      test_team_i, test_team_j, omega)

        return fold_num, test_cost_nll, train_theta_hat

    # ---- Main loop ----
    for lam_idx, lam in enumerate(lambdas_sorted):

        results_parallel = Parallel(n_jobs=-1, backend="loky")( delayed(process_fold)
                                    (fold_num, train_idx, test_idx, lam, fold_theta_inits[fold_num])
                                    for fold_num, (train_idx, test_idx) in enumerate(fold_indices) )

        loc_nll = []

        for fold_num, test_cost_nll, train_theta_hat in results_parallel:
            loc_nll.append(test_cost_nll)

            if use_hot_start and lam_idx < len(lambdas_sorted) - 1:
                fold_theta_inits[fold_num] = train_theta_hat.copy()

        avg_nll.append(np.mean(loc_nll))

    avg_nll = np.array(avg_nll)
    results = np.column_stack((lambdas[lambda_sort_idx], avg_nll[lambda_sort_idx]))

    min_idx = np.argmin(results[:, 1])
    OptimumVal = np.array([results[min_idx, 0], results[min_idx, 1]])

    return {'Results': results,
            'OptimalVal': OptimumVal,
            'Opt_lambda': float(results[min_idx, 0]),
            'fold_assignments': fold_info }


# ####### ==========================================================
# ####### -------- Performing cross validation in serial -----------
# ####### ==========================================================
# def fit_logistic_cv_model(df, lambdas, omega, n_splits, seedNum=None, mu0=None, 
#                           fold_assignments=None, use_hot_start=True, real_data = False):
#     """
#     This function performs CV in serial (a bit slower compared to the parallel implementation)
#     Perform k-fold cross-validation to find the optimal regularization parameter.
    
#     Parameters
#     ----------------------
#     df       : pd.DataFrame (Match data with team sizes and elimination outcomes)
#     lambdas  : array-like  (Sequence of lambda values to test)
#     omega    : array-like of length 2 ( Weights [elimination_weight, pairwise_weight])
#     n_splits : int, optional ( Number of folds for cross-validation (default: 6))
#                Ignored if fold_assignments is provided
#     seedNum  : int, optional ( Random seed for reproducibility (default: 35))
#                Ignored if fold_assignments is provided
#     mu0      : array-like, optional
#                Initial parameter values (default: survival-based for lambda=0)
#                If None, uses survival rankings for lambda=0, then hot start for others
#     fold_assignments : dict, optional
#                 Pre-generated fold assignments from generate_cv_folds()
#                 If provided, uses these instead of generating new folds
#     use_hot_start : bool, optional
#                 If True, use solution from previous lambda as initialization (default: True)
#                 Requires lambdas to be sorted (smallest to largest recommended)
    
#     Returns
#     -----------------------
#     dict
#         'Results': (n_lambdas, 2) array of [lambda, avg_nll]
#         'OptimalVal': [optimal_lambda, min_avg_nll]
#         'Opt_lambda': optimal lambda value
#         'CV_Details': list of nll values for each fold and lambda
#         'fold_assignments': The fold assignments used (for inspection)
#     """

#     lambdas = np.asarray(lambdas, dtype=float).ravel()
#     omega = np.asarray(omega, dtype=float).ravel()

#     if omega.size != 2:
#         raise ValueError("omega must have length 2")
    
#     n_splits = int(n_splits)

#     if seedNum is not None:
#         seedNum = int(seedNum)
    
#     ## Sort lambdas for hot start (smallest to largest)
#     if use_hot_start:
#         lambda_sort_idx = np.argsort(lambdas)
#         lambdas_sorted = lambdas[lambda_sort_idx]
#     else:
#         lambdas_sorted = lambdas
#         lambda_sort_idx = np.arange(len(lambdas))
    
#     ## Note: data_type = True means using the actual game data
#     if real_data is False:
#         n_cols = [col for col in df.columns if str(col).startswith("n_")]
#         matN = to_numpy(df[n_cols])
#     else: 
#         matN = to_numpy(df.iloc[:,11:-2])     # When using actual game data
    
#     _, N = matN.shape
#     B = build_B_sum_to_zero(N)
    
#     ## Smart initialization based on survival rankings
#     if mu0 is None:
#         # Use survival-based initialization for first lambda (typically lambda=0)
#         mu0 = compute_initial_mu(matN)
#         mu0 = to_numpy(mu0).ravel()
#     else:
#         ## Ensure mu0 is proper numpy array if provided from R
#         mu0 = np.asarray(mu0, dtype=float).ravel()

#     theta0 = np.concatenate(([1.0, 1.0], mu0)).flatten()
    
#     if len(mu0) != N - 1:
#         raise ValueError(f"mu0 must be of length N-1 (got {len(mu0)}, expected {N-1})")
    
#     ## Use provided fold assignments or generate new ones
#     if fold_assignments is not None:
#         ## Use pre-generated fold assignments
#         fold_indices = fold_assignments['fold_indices']
#         n_splits = fold_assignments['n_splits']
#         fold_info = fold_assignments  # Save for return
#     else:
#         # Generate fold assignments
#         fold_info = generate_cv_folds(df, n_splits=n_splits, seedNum=seedNum)
#         fold_indices = fold_info['fold_indices']
    
#     lambda0 = 0.0  # Unpenalized for testing
    
#     avg_nll = []        # Store average nll score for each lambda value
    
#     ## Initialize fold-specific starting values ONCE (outside lambda loop)
#     ## For hot start, these will be updated after each lambda
#     fold_theta_inits = [theta0.copy() for _ in range(len(fold_indices))]
    
#     ## Hot start - loop over sorted lambdas
#     for lam_idx, lam in enumerate(lambdas_sorted):
#         loc_nll = []    # Store nll for each fold
        
#         for fold_num, (train_idx, test_idx) in enumerate(fold_indices):
#             df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        
#             ## Note: real_data = True means using the actual game data
#             if real_data is False:
#                 ## ---- Using simulated data -----
#                 n_cols = [col for col in df_train.columns if str(col).startswith("n_")]
#                 train_matN = to_numpy(df_train[n_cols])
#                 train_team_i = to_numpy(df_train["Team i"]).astype(int)
#                 train_team_j = to_numpy(df_train["Team j"]).astype(int)
#                 train_y = to_numpy(df_train["Elimination"]).astype(int)
                
#                     ### Testing data
#                 n_cols = [col for col in df_test.columns if str(col).startswith("n_")]
#                 test_matN = to_numpy(df_test[n_cols])
#                 test_team_i = to_numpy(df_test["Team i"]).astype(int)
#                 test_team_j = to_numpy(df_test["Team j"]).astype(int)
#                 test_y = to_numpy(df_test["Elimination"]).astype(int)
#             else:
#                 ## ---- Using actual game data -----
#                 train_matN = to_numpy(df_train.iloc[:,11:-2])
#                 train_team_i = to_numpy(df_train["Team i (ID)"]).astype(int)
#                 train_team_j = to_numpy(df_train["Team j (ID)"]).astype(int)
#                 train_y = to_numpy(df_train["Elimination"]).astype(int)
                
#                 ### Testing data
#                 test_matN = to_numpy(df_test.iloc[:,11:-2])
#                 test_team_i = to_numpy(df_test["Team i (ID)"]).astype(int)
#                 test_team_j = to_numpy(df_test["Team j (ID)"]).astype(int)
#                 test_y = to_numpy(df_test["Elimination"]).astype(int)
                
            
            
#             ## MOD: Use fold-specific initialization (hot start from previous lambda)
#             current_init = fold_theta_inits[fold_num]
            
#             ## Fitting the model using optimization
#             train_res = minimize( fun=comb_nll_only, x0=current_init, jac=comb_grad_nll_only,
#                                   args=(lam, train_matN, train_y, train_team_i, train_team_j, omega),
#                                   method='BFGS', options={"gtol": 1e-8, "maxiter": 250, "disp": False} )
            
#             # Recover optimal values
#             train_theta_hat = train_res.x
        
#             ## Save solution for hot start in next lambda iteration
#             if use_hot_start and lam_idx < len(lambdas_sorted) - 1:
#                 fold_theta_inits[fold_num] = train_theta_hat.copy()
             
#             # Testing: Evaluate unpenalized log-likelihood on test set
#             test_cost_nll = comb_nll_only( train_theta_hat, lambda0, test_matN, test_y, 
#                                           test_team_i, test_team_j, omega )
            
#             loc_nll.append(test_cost_nll)
        
#         avg_nll.append(np.mean(loc_nll))

#     avg_nll = np.array(avg_nll)
#     results = np.column_stack((lambdas[lambda_sort_idx], avg_nll[lambda_sort_idx]))    # Result matrix.

#     min_idx = np.argmin(results[:, 1]) 
#     OptimumVal = np.array([results[min_idx, 0], results[min_idx, 1]])
#     OptLambda = results[min_idx, 0]
    
#     ## MOD: Convert to native Python types for R compatibility
#     return {
#         'Results': results,
#         'OptimalVal': OptimumVal,
#         'Opt_lambda': float(OptLambda),
#         'fold_assignments': fold_info       # Return fold assignments for inspection
#     }


# ####### ==================================================================
# ####### -------------- Out put -------------------------------------------
# ####### ==================================================================
# '''
# - Uncomment the part below, ensure the code is working before you proceed to other function files
# - Keep the lower part commented when running other function files that depend on this
# '''
# StartTime = time.perf_counter()

# NumOfTeams = 16                   ## Number of teams in the tournament
# n0 = 4                            ## Initial team size (solo = 1, dua = 2, squad = 4)

# elim_omega = 0.50                  ## weight assigned to elimination model
# pw_omega = 1 - elim_omega         ## weight assigned to pairwise model
# omega = [elim_omega, pw_omega]    ## weight vector

# seedNum = 125                     ## seed number for reproducibility

# alpha_max = 2.0                   ## maximum value of team strength

# n_splits = 6                      ## number of split for cross validation

# Np = 100                          ## number of partition for the lambda domain
# lam0 = 0.0                        ## lambda value for unpenalized model
# marker_spacing = int(np.ceil(Np/20))         ## marker spacing when ploting 

# ## lambdas = np.linspace(0,10, Np)            ## used for uniform partition of the lambda domain
# lambdas = np.logspace(-8, np.log10(10), Np)  ## used for logarithmic partition of the lambda domain

# num_of_stack_data = 5              ## number of stacked matches
# real_data = False                  ## selecting either real or simulated match data
# elimination_data= True             ## selecting either elimination or pairwise data


# ###======= Getting match data ==========================================
# ## ----- Impoting real match data: comment if real_data = False --------
# # df = pd.read_csv('Match 1.csv').dropna()

# ## ----- Simulating match data: comment if real_data = True  -----------
# alpha = generate_alpha(NumOfTeams, alpha_max, seedNum)
# df = combined_match_data(alpha, n0, num_of_stack_data, elimination_data=True, data_rand_state=None)

# ##### ---- fitting the model and performing cross validation ----------
# result = comb_fit_logistic_model(df, lam0, omega, NumOfTeams, seedNum, alpha_max, mu0=None, real_data = real_data)

# cv_res = fit_logistic_cv_model(df, lambdas, omega, n_splits, seedNum=seedNum, mu0=None, 
#                                fold_assignments=None, use_hot_start=True, real_data=real_data)

# EndTime = time.perf_counter()
# print(f"Elapse time : {EndTime - StartTime:.5f} seconds")

# ##### *********************************************************************
# ###### ------- Printing a table of various rankings ------------------------
# print(tabulate(result['Predicted Ranking'], headers='keys', tablefmt='pretty', showindex=False))

# print('Ranking RMSE = ', result['Predicted RMSE'], ', Ranking MAE = ', result["Predicted MAE"]
#       ,', RankingSpearmans = ', result['Predicted Spearmans'], ', Perfect ranking = ', result['Perfect prediction'])


# print(tabulate(result['Team size Ranking'], headers='keys', tablefmt='pretty', showindex=False))

# print('Survival RMSE = ', result['Team size RMSE'], ', Survival MAE = ', result["Team size MAE"]
#       ,', Survival Spearmans = ', result['Team size Spearmans'], ', Surv. perfect = ', result['Team size perfect_pred'],)


# print('Strength recover RMSE', result['RMSE between alpha'], 'Strength recover MAE', result['MAE between alpha'])

# print("NLL = ", result["Opt_lik_est"], 'beta = ', result['beta_hat'], 'gamma = ', result['gamma_hat'])


# ###### *********************************************************************
# ###### ------- Plotting the cross validation curve -------------------------
# opt_lambda = cv_res['Opt_lambda']
# lambdaVals = cv_res['Results'][:,0]
# NLL_vals = cv_res['Results'][:,1]

# plt.figure(figsize=(8, 4))
# plt.plot(lambdaVals, NLL_vals, marker='o', linestyle='-.', markevery=marker_spacing, color='b')
# plt.axvline(x=opt_lambda, color='m', linestyle='--', label=f'Optimal $\\lambda$ = {opt_lambda:.3f}' )

# plt.xscale('log')
# ymin, ymax = plt.ylim()
# ticks = np.logspace(np.log10(ymin), np.log10(ymax), 8)
# plt.yticks(ticks, [f"{t:.2f}" for t in ticks])
# plt.minorticks_off()

# plt.xlabel('Ridge penalty parameter $\\lambda$',  fontweight='bold', fontsize=13 )
# plt.ylabel('Estimated NLL',  fontweight='bold', fontsize=13)
# plt.title( f'CV Curve for, Optimal $\\lambda$ = {opt_lambda:.3f}',  fontweight='bold', fontsize=13)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"CV_curve_.jpeg", dpi=300)
# plt.show()




