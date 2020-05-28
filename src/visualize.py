# File contains functions to visualize data, analyze results and to be reusable through 
# our data science process
import random

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, make_scorer
)
import seaborn as sns
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx


import pandas as pd 
import numpy as np


def plot_seasonal_decompose(series, model='additive', freq=None):

    fig, ax = plt.subplots(4, 1, figsize=(25, 15))

    result = seasonal_decompose(series, model=model, freq=freq)
    result.observed.plot(ax=ax[0])
    result.trend.plot(ax=ax[1])
    result.seasonal.plot(ax=ax[2])
    result.resid.plot(ax=ax[3])

    ax[0].set_ylabel("Original Series")
    ax[1].set_ylabel("Trend-Cycle Series")
    ax[2].set_ylabel("Seasonal Series")
    ax[3].set_ylabel("Residual Series")
    
    return result

def tsplot(y, lags=None, figsize=(10, 8), **kargs):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, **kargs)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    sns.despine()
    plt.tight_layout()
    
    return ts_ax, acf_ax, pacf_ax

def plot_series(df, title=None):
    fig = go.Figure([go.Scatter(x=df.reset_index()['data_da_operacao'], 
                                y=df.reset_index()['qtd_premio'])])

    fig.update_layout(title_text=title,
                      xaxis_rangeslider_visible=True)

    return fig


def collect_reg_metrics(true, pred):
    r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)

    return {"R2": r2,
            "MSE": mse, 
            "RMSE": rmse, 
            "MAE": mae}

def create_fitted_df(dates, true, pred):
    fitted_df =  pd.DataFrame({'date': dates,
                                'pred': pred.reshape(1, -1)[0],
                                'true': true.reshape(1, -1)[0]})

    fitted_df['residuals'] = (fitted_df.true - fitted_df.pred)

    return fitted_df


def check_residuals(fitted_df, **kargs):
    _, ax1 = plt.subplots(figsize=(16, 10))
    _, ax2 = plt.subplots(1, 2, figsize=(16, 10))

    sns.lineplot(x='date',
                 y='residuals',
                 data=fitted_df,
                 ax=ax1)

    smt.graphics.plot_acf(fitted_df['residuals'],
                        lags=90,
                        ax=ax2[0])

    sns.distplot(fitted_df.residuals.values, 
                 rug=True, 
                 hist=True,
                 kde=True,
                 ax=ax2[1])

    ax1.set_title("Residual Diagnostic Plots")
    ax2[0].set_xlabel("Lag")
    ax2[1].set_xlabel("Residual")

def hierarchy_pos(G, 
                  root=None, 
                  width=1., 
                  vert_gap=0.2, 
                  vert_loc=0, 
                  xcenter=0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    G: the graph (must be a tree)
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, 
                      root, 
                      width=1., 
                      vert_gap=0.2, 
                      vert_loc=0,
                      xcenter=0.5, 
                      pos=None, 
                      parent=None):
        '''
        see hierarchy_pos docstring for most arguments
        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def make_annotations(pos, font_size=10, font_color='rgb(250,250,250)', M=None):
    annotations = []
    for label, pos in pos.items():
        annotations.append(
            dict(
                text=label,
                x=pos[0], y=2*M+pos[1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations

def get_nodes_edges_position(edges, root="total", **kargs):

    G = nx.Graph()
    G.add_edges_from(edges)

    positions = hierarchy_pos(G, root=root, width=10000)
    positions = {key:list(value) for key, value in positions.items()}

    nodes_x = [position[0] for position in positions.values()]
    nodes_y = [position[1] for position in positions.values()]

    M = max(nodes_y)
    edges_x = []
    edges_y = []

    for edge in edges:
        edges_x += [positions[edge[0]][0],positions[edge[1]][0], None]
        edges_y += [10*M+positions[edge[0]][1],10*M+positions[edge[1]][1], None]

    labels = list(positions.keys())
    annotations = make_annotations(positions, M=M)

    return nodes_x, nodes_y, edges_x, edges_y, labels, annotations
