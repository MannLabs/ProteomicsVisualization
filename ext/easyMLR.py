import numpy as np
import pandas as pd
import numba as nb
#import numba_stats as nbs
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats
#import statsmodels.api as sm
from collections import Counter
import multiprocessing
from joblib import Parallel, delayed
import random
import math
#import swifter
from sklearn.preprocessing import StandardScaler

### Background ###################################################################
# This script contains functions to perform two-sample t-tests with s0 correction
# and a permutation-based FDR model.
# The strategy has been implemented on the basis following previous work:
# 1) Tusher VG, Tibshirani R, Chu G. Significance analysis of microarrays applied
# to the ionizing radiation response. Proc Natl Acad Sci U S A 2001;98:5116–21.
# https://doi.org/10.1073/pnas.091062498.
# 2) Tyanova S, Cox J. Perseus: A bioinformatics platform for integrative analysis
# of proteomics data in cancer research. Methods Mol. Biol., vol. 1711,
# Humana Press Inc.; 2018, p. 133–48. https://doi.org/10.1007/978-1-4939-7493-1_7.
##################################################################################

def get_std(x):
    """
    Function to calculate the sample standard deviation.
    """
    std_x = np.sqrt(np.sum((abs(x - x.mean())**2)/(len(x)-1)))
    return std_x

def perform_ttest(x, y, s0):
    """
    Function to perform a independent two-sample t-test including s0 adjustment.
    Assumptions: Equal or unequal sample sizes with similar variances.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Get fold-change
    fc = mean_x-mean_y

    n_x = len(x)
    n_y = len(y)

    # pooled standard vatiation
    # assumes that the two distributions have the same variance
    sp = np.sqrt((((n_x-1)*get_std(x)**2) + ((n_y-1)*(get_std(y)**2)))/(n_x+n_y-2))

    # Get t-values
    tval = fc/(sp * (np.sqrt((1/n_x)+(1/n_y))))
    tval_s0 = fc/((sp * (np.sqrt((1/n_x)+(1/n_y))))+s0)

    # Get p-values
    pval =2*(1-stats.t.cdf(np.abs(tval),n_x+n_y-2))
    pval_s0 =2*(1-stats.t.cdf(np.abs(tval_s0),n_x+n_y-2))

    return [fc, tval, pval, tval_s0, pval_s0]

def generate_perms(n, n_rand, seed = 44):
    """
    Generate n_rand permutations of indeces ranging from 0 to n.
    """
    np.random.seed(seed)
    idx_v = np.arange(0,n)
    rand_v = list()
    n_rand_i = 0
    n_rand_max = math.factorial(n)-1
    if n_rand_max <= n_rand:
        print("{} random permutations cannot be created. The maximum of n_rand={} is used instead.".format(n_rand, n_rand_max))
        n_rand = n_rand_max
    while n_rand_i < n_rand:
        rand_i = list(np.random.permutation(idx_v))
        if np.all(rand_i == idx_v):
            next
        else:
            if rand_i in rand_v:
                next
            else:
                rand_v.append(rand_i)
                n_rand_i = len(rand_v)
    return rand_v

def permutate_vars(X, n_rand, seed = 44):
    """
    Function to randomly permutate an array `n_rand` times.
    """
    x_perm_idx = generate_perms(n=len(X), n_rand=n_rand, seed = seed)

    X_rand = list()
    for i in np.arange(0,len(x_perm_idx)):

        X_rand.append([X[j] for j in x_perm_idx[i]])

    return X_rand

def workflow_ttest(df, c1, c2, s0=1, parallelize=False):
    """
    Function to perform a t-test on all rows in a data frame.
    c1 specifies the column names with intensity values of the first condition.
    c2 specifies the column names with intensity values of the second condition.
    s0 is the tuning parameter that specifies the minimum fold-change to be trusted.
    """
    if parallelize:
        res = df.swifter.apply(lambda row : perform_ttest(row[c1], row[c2], s0=s0), axis = 1)
    else:
        res = df.apply(lambda row : perform_ttest(row[c1], row[c2], s0=s0), axis = 1)

    res = pd.DataFrame(list(res), columns=['fc','tval','pval','tval_s0','pval_s0'])

    df_real = pd.concat([df, res], axis=1)

    return df_real

def workflow_permutation_tvals(df, c1, c2, s0=1, n_perm=2, parallelize=False):
    """
    Function to perform a t-test on all rows in a data frame based on the permutation of
    samples across conditions.
    c1 specifies the column names with intensity value sof the first condition.
    c2 specifies the column names with intensity value sof the second condition.
    s0 is the tuning parameter that specifies the minimum fold-change to be trusted.
    n_perm specifies the number of random permutations to perform.
    """
    all_c = c1 + c2
    all_c_rand = permutate_vars(all_c, n_rand=n_perm)

    res_perm = list()
    for i in np.arange(0,len(all_c_rand)):
        if parallelize:
            res_i = df.swifter.apply(lambda row : perform_ttest(row[all_c_rand[i][0:len(c1)]],
                                                                row[all_c_rand[i][len(c1):len(c1)+len(c2)]],
                                                                s0=0.1),
                                                                axis = 1)
        else:
            res_i = df.apply(lambda row : perform_ttest(row[all_c_rand[i][0:len(c1)]],
                                                        row[all_c_rand[i][len(c1):len(c1)+len(c2)]],
                                                        s0=0.1),
                                                        axis = 1)

        res_i = pd.DataFrame(list(res_i), columns=['fc','tval','pval','tval_s0','pval_s0'])
        res_perm.append(list(np.sort(np.abs(res_i.tval_s0.values))))

    return res_perm

def get_tstat_cutoff(res_real, t_perm_avg, delta):
    """
    Extract the minimum t-stat value for which the difference
    between the observed and average random t-stat is
    larger than the selected delta.
    """
    # Extract all t-stats from the results of the 'real' data
    # and sort the absolute values.
    t_real = res_real.tval_s0
    t_real_abs = np.sort(np.abs(t_real))
    # print(t_real_abs)

    # Calculate the difference between the observed t-stat
    # and the average random t-stat for each rank position.
    t_diff = t_real_abs - t_perm_avg
    # print(t_diff)

    # Find the minimum t-stat value for which the difference
    # between the observed and average random t-stat is
    # larger than the selected delta.
    t_max = t_real_abs[t_diff > delta]
    #print(t_max.shape[0])
    if (t_max.shape[0] == 0):
        t_max = np.ceil(np.max(t_real_abs))
    else:
        t_max = np.min(t_max)

    #print(t_max)

    return t_max

@nb.njit
def get_positive_count(res_real_tval_s0, t_cut):
    """
    Count number of tval_s0 in res_real that are above the t_cut threshold.
    """
    ts_sig = list()
    for r in res_real_tval_s0:
        r_abs = np.abs(r)
        if r_abs >= t_cut:
            ts_sig.append(r_abs)

    return len(ts_sig)

@nb.njit
def get_false_positive_count(res_perm, t_cut):
    """
    Get median number of tval_s0 in res_perm that are above the t_cut threshold.
    """
    ts_sig = list()
    for rp in res_perm:
        ts_abs = np.abs(rp)
        ts_max = list()
        for t in ts_abs:
            if t >= t_cut:
                ts_max.append(t)

        ts_sig.append(len(ts_max))

    res = np.median(np.array(ts_sig))

    return res

def get_pi0(res_real, res_perm):
    """
    Estimate pi0, the proportion of true null (unaffected) genes in the data set, as follows:
    (a) Compute q25; q75 = 25% and 75% points of the permuted d values (if p = # genes,
    B = # permutations, there are pB such d values).
    (b) Count the number of values d in the real dataset that are between q25 and q75
    (there are p values to test). pi0 = #d/0.5*p.
    (c) Let pi0 = min(pi0; 1) (i.e., truncate at 1).
    Documentation in: https://statweb.stanford.edu/~tibs/SAM/sam.pdf
    """
    t_real = res_real["tval_s0"]
    t_real_abs = np.sort(np.abs(t_real))

    t_perm = list()
    for ri in res_perm:
        for r in ri:
            t_perm.append(r)
    t_perm_abs = np.sort(np.abs(t_perm))

    t_perm_25 = np.percentile(t_perm_abs, 25)
    t_perm_75 = np.percentile(t_perm_abs, 75)

    n_real_in_range = np.sum((t_real_abs >= t_perm_25) & (t_real_abs <= t_perm_75))
    pi0 = n_real_in_range/(0.5*len(t_real_abs))

    # pi0 can maximally be 1
    pi0 = np.min([pi0, 1])

    return pi0

def get_fdr(n_pos, n_false_pos, pi0):
    """
    Compute the FDR by dividing the number of false positives by the number of true positives.
    The number of false positives are adjusted by pi0, the proportion of true null (unaffected)
    genes in the data set.
    """
    n = n_false_pos*pi0
    if n != 0:
        if n_pos != 0:
            fdr = n/n_pos
        else:
            fdr = 0
    else:
        if n_pos > 0:
            fdr = 0
        else:
            fdr = np.nan
    return fdr

def estimate_fdr_stats(res_real, res_perm, delta):
    """
    Helper function for get_fdr_stats_across_deltas.
    It computes the FDR and tval_s0 thresholds for a specified delta.
    The function returns a list of the following values:
    t_cut, n_pos, n_false_pos, pi0, n_false_pos_corr, fdr
    """
    perm_avg = np.mean(res_perm, axis=0)
    t_cut = get_tstat_cutoff(res_real, perm_avg, delta)
    n_pos = get_positive_count(res_real_tval_s0=np.array(res_real.tval_s0), t_cut=t_cut)
    n_false_pos = get_false_positive_count(np.array(res_perm), t_cut=t_cut)
    pi0 = get_pi0(res_real, res_perm)
    n_false_pos_corr = n_false_pos*pi0
    fdr = get_fdr(n_pos, n_false_pos, pi0)
    return [t_cut, n_pos, n_false_pos, pi0, n_false_pos_corr, fdr]

def get_fdr_stats_across_deltas(res_real, res_perm):
    """
    Wrapper function that starts with the res_real and res_perm to derive
    the FDR and tval_s0 thresholds for a range of different deltas.
    """
    res_stats = list()
    for d in np.arange(0,10,0.01):
        res_d = estimate_fdr_stats(res_real, res_perm, d)
        if np.isnan(res_d[5]):
            break
        else:
            res_stats.append(res_d)
    res_stats_df = pd.DataFrame(res_stats,
                                columns=['t_cut', 'n_pos', 'n_false_pos', 'pi0', 'n_false_pos_corr', 'fdr'])
    return(res_stats_df)

def get_tstat_limit(stats, fdr=0.01):
    """
    Function to get tval_s0 at the specified FDR.
    """
    t_limit = np.min(stats[stats.fdr<=fdr].t_cut)
    return(t_limit)

def annotate_fdr_significance(res_real, stats, fdr=0.01):
    t_limit = np.min(stats[stats.fdr<=fdr].t_cut)
    res_real['qval'] = [np.min(stats[stats.t_cut<=abs(x)].fdr) for x in res_real['tval_s0']]
    res_real['FDR ' + str(int(fdr*100)) + '%'] = ["sig" if abs(x)>=t_limit else "non_sig" for x in res_real['tval_s0']]
    return(res_real)

def perform_ttest_getMaxS(fc, s, s0, n_x, n_y):
    """
    Helper function to get ttest stats for specified standard errors s.
    Called from within get_fdr_line.
    """
    # Get t-values
    tval = fc/s
    tval_s0 = fc/(s+s0)

    # Get p-values
    pval =2*(1-stats.t.cdf(np.abs(tval),n_x+n_y-2))
    pval_s0 =2*(1-stats.t.cdf(np.abs(tval_s0),n_x+n_y-2))

    return [fc, tval, pval, tval_s0, pval_s0]

def get_fdr_line(t_limit, s0, n_x, n_y, plot = False,
                 fc_s = np.arange(0,6,0.01), s_s = np.arange(0.005,6,0.005)):
    """
    Function to get the fdr line for a volcano plot as specified tval_s0 limit, s0, n_x and n_y.
    """
    pvals = np.ones(len(fc_s))
    svals = np.zeros(len(fc_s))
    for i in np.arange(0,len(fc_s)):
        for j in np.arange(0,len(s_s)):
            res_s = perform_ttest_getMaxS(fc=fc_s[i], s=s_s[j], s0=s0, n_x=n_x, n_y=n_y)
            if res_s[3] >= t_limit:
                if svals[i] < s_s[j]:
                    svals[i] = s_s[j]
                    pvals[i] = res_s[2]

    s_df = pd.DataFrame(np.array([fc_s,svals,pvals]).T, columns=['fc_s','svals','pvals'])
    s_df = s_df[s_df.pvals != 1]

    s_df_neg = s_df.copy()
    s_df_neg.fc_s = -s_df_neg.fc_s

    s_df = s_df.append(s_df_neg)

    if (plot):
        fig = px.scatter(x=s_df.fc_s,
                         y=-np.log10(s_df.pvals),
                         template='simple_white')
        fig.show()

    return(s_df)


def plot_volcano(res_real, fdr_line, t_limit, id_col='PG.Genes', show=True):
    """
    Plot a volcano plot with FDR lines.
    """
    res_plot = res_real.copy()
    res_plot['tlim_sig'] = ["sig" if abs(x)>=t_limit else "non_sig" for x in res_plot['tval_s0']]
    fig = px.scatter(x=res_plot.fc,
                     y=-np.log10(res_plot.pval),
                     color=res_plot.tlim_sig,
                     hover_name=res_plot[id_col],
                     template='simple_white', render_mode="svg",
                     labels=dict(x="log2 fold change", y="-log10(p-value)",
                                 color=[col for col in res_plot.columns if col.startswith("FDR ")][0]
                                )
                    )
    if fdr_line is not None:
        fig.add_trace(go.Scatter(x=fdr_line[fdr_line.fc_s > 0].fc_s,
                                 y=-np.log10(fdr_line[fdr_line.fc_s > 0].pvals),
                                 line_color="black",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=fdr_line[fdr_line.fc_s < 0].fc_s,
                                 y=-np.log10(fdr_line[fdr_line.fc_s < 0].pvals),
                                 line_color="black",
                                 showlegend=False))

    fig.update_layout(width=600, height=700)

    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': "easyMLR_volcano"
        }
    }

    if show:
        fig.show(config=config)
    return fig

def perform_ttest_analysis(df, c1, c2, s0=1, n_perm=2, fdr=0.01, id_col='Genes', plot_fdr_line=False, parallelize=False, show=False):
    """
    Workflow function for the entire T-test analysis including FDR
    estimation and visualizing a volcanoe plot.
    """
    ttest_res = workflow_ttest(df, c1, c2, s0, parallelize=parallelize)
    ttest_perm_res = workflow_permutation_tvals(df, c1, c2, s0, n_perm, parallelize=parallelize)
    ttest_stats = get_fdr_stats_across_deltas(ttest_res, ttest_perm_res)
    ttest_res = annotate_fdr_significance(res_real=ttest_res, stats=ttest_stats, fdr=fdr)
    t_limit = get_tstat_limit(stats=ttest_stats, fdr=fdr)
    if plot_fdr_line:
        fc_max = np.max(np.abs(ttest_res.fc))
        fc_itr = fc_max/100
        fdr_line = get_fdr_line(t_limit=t_limit, s0=s0, n_x=len(c1), n_y=len(c2), fc_s = np.arange(0,fc_max,fc_itr), plot=False) #adjust
    else:
        fdr_line = None
    fig = plot_volcano(res_real=ttest_res, fdr_line=fdr_line, t_limit=t_limit, id_col=id_col, show=show)
    return(ttest_res, fig)
