# -*- coding: utf-8 -*-
"""
Created on 2024-05-09 (Thu) 14:22:39

Utils for benchmarking of skeleton estimation methods.

@author: I.Azuma
"""
# %%
import copy
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error

def g2array(g):
    return nx.adjacency_matrix(g).todense()

def pair_melt(orig_df):
    """
    Args:
        df (dataframe): Adjacency matrix
                            Hepatocyte  Hepatoblast  Cholangiocyte
        Hepatocyte            0.0     1.683933            0.0
        Hepatoblast           0.0     0.000000            0.0
        Cholangiocyte         0.0     1.549065            0.0

    Returns:
        dataframe: Melted dataframe as follows:
                Source         Target     value
        8  Cholangiocyte  Cholangiocyte  0.000000
        5  Cholangiocyte    Hepatoblast  1.549065
        2  Cholangiocyte     Hepatocyte  0.000000
        7    Hepatoblast  Cholangiocyte  0.000000
        4    Hepatoblast    Hepatoblast  0.000000
        1    Hepatoblast     Hepatocyte  0.000000
        6     Hepatocyte  Cholangiocyte  0.000000
        3     Hepatocyte    Hepatoblast  1.683933
        0     Hepatocyte     Hepatocyte  0.000000
    """
    df = copy.deepcopy(orig_df)
    df['Source']=df.index.tolist()
    df_melt = df.melt(id_vars='Source',var_name='Target').sort_values('value')
    df_melt = df_melt.sort_values(['Source','Target'])
    
    return df_melt

def convert2binary(df):
    mat = np.array(df)
    for i in range(mat.shape[0]):
        for j in range(i+1,mat.shape[1]):
            if mat[i][j] == mat[j][i]:
                mat[i][j] = 0
                mat[j][i] = 0
            elif mat[i][j] > mat[j][i]:
                mat[i][j] = 1
                mat[j][i] = 0
            else:
                mat[j][i] = 1
                mat[i][j] = 0
        mat[i][i] = 0  # Convert the diagonal component to 0.
    binary_df = pd.DataFrame(mat,index=df.index,columns=df.columns)
    return binary_df

def triu2flatten(df):
    mat = np.array(df)
    res = []
    st_list = []
    for i in range(mat.shape[0]):
        for j in range(i+1,mat.shape[1]):
            res.append(mat[i][j])
            st_list.append((df.index.tolist()[i], df.columns.tolist()[j]))
    return res, st_list

def mat2flatten(df):
    mat = np.array(df)
    res = []
    st_list = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j:
                pass
            else:
                res.append(mat[i][j])
                st_list.append((df.index.tolist()[i], df.columns.tolist()[j]))
    return res, st_list

def eval_with_w(adj_df,W,do_abs=True,title="[]",weight_threshold=0.3,common=[]):
    res_range = adj_df.index.tolist()
    W_pro = pd.DataFrame(np.zeros((adj_df.shape[1], adj_df.shape[1])),index=res_range,columns=res_range)
    if len(common) == 0:
        common = list(set(res_range) & set(W.index))
    for c1 in common:
        for c2 in common:
            if c1 == c2:  # Ignore diagonal components
                pass
            else:
                if do_abs:
                    W_pro[c1].loc[c2] = abs(W[c1].loc[c2])  # abs
                else:
                    W_pro[c1].loc[c2] = W[c1].loc[c2]

    # minmax scaling
    vmax = W_pro.max().max()
    vmin = W_pro.min().min()
    fxn = lambda x : (x-vmin)/(vmax-vmin)
    mm_W = W_pro.applymap(fxn)

    # intersecion cell types
    common = sorted(common)
    adj_common = adj_df[common].loc[common]
    W_common = mm_W[common].loc[common]
    adj_melt = pair_melt(adj_common)
    w_melt = pair_melt(W_common)

    fig, ax = plt.subplots(1,2,figsize=(15,7))
    sns.heatmap(W_common,cmap='Oranges',ax=ax[0])
    sns.heatmap(adj_common,cmap='Purples',ax=ax[1])
    plt.show()

    # scatter plots for overall relationships
    y_true = w_melt['value'].tolist()
    y_score = adj_melt['value'].tolist()
    total_cor, pvalue = stats.pearsonr(y_score,y_true)
    rmse = np.sqrt(mean_squared_error(y_score, y_true))

    # scatter plots for dag positive edges
    y_true_posi = []
    y_score_posi = []
    c = []
    for i in range(len(y_score)):
        if abs(y_score[i])>weight_threshold:
            c.append('tab:orange')
            y_score_posi.append(y_score[i])
            y_true_posi.append(y_true[i])
        else:
            c.append('tab:blue')

    fig,ax = plt.subplots(figsize=(5,4))
    plt.scatter(y_true,y_score,color=c)
    plt.xlabel('sc RNA-Seq CCC Reference')
    plt.ylabel('Estimated Adj Weight')
    plt.text(1.0,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes)
    plt.text(1.0,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.title(title)
    plt.show()

    # scatterplots with distribution
    plot_sns_scatter(y_true_posi, y_score_posi, c="tab:orange")


# %% Visualization
def plot_scatter(y_true:list, y_score:list, xlabel="sc RNA-Seq CCC Reference",ylabel="Estimated Adj Weight",title="",c="tab:blue"):
    assert len(y_true)==len(y_score), "! y_true and y_score must be the same length !"

    total_cor, pvalue = stats.pearsonr(y_score,y_true)  # Pearson correlation
    rmse = np.sqrt(mean_squared_error(y_score, y_true))  # RMSE
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)

    fig,ax = plt.subplots(figsize=(5,4))
    plt.scatter(y_true,y_score,color=c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(.8,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes)
    plt.text(.8,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes)
    plt.text(.8,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.title(title)
    plt.show()

def plot_sns_scatter(y_true:list, y_score:list, xlabel="sc RNA-Seq CCC Reference",ylabel="Estimated Adj Weight",title="",c="tab:blue"):
    assert len(y_true)==len(y_score), "! y_true and y_score must be the same length !"

    total_cor, pvalue = stats.pearsonr(y_score,y_true)  # Pearson correlation
    rmse = np.sqrt(mean_squared_error(y_score, y_true))  # RMSE
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)

    tmp_df = pd.DataFrame({"y_true":y_true,"y_score":y_score})
    
    ax = sns.jointplot(data=tmp_df,x="y_true",y="y_score",kind="reg",color=c)
    xmax, xmin = max(y_true), min(y_true)
    ymax, ymin = max(y_score), min(y_score)

    plt.text(xmin+.8*(xmax-xmin),ymin+0.35*(ymax-ymin),'R = {}'.format(str(round(total_cor,3))))
    plt.text(xmin+.8*(xmax-xmin),ymin+0.30*(ymax-ymin),'P = {}'.format(str(pvalue)))
    plt.text(xmin+.8*(xmax-xmin),ymin+0.25*(ymax-ymin),'RMSE = {}'.format(str(round(rmse,3))))
    plt.xlabel("sc RNA-Seq CCC Reference")
    plt.ylabel("Estimated Adj Weight")
    plt.title(title)
    plt.show()
