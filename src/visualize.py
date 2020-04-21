# File contains functions to visualize data, analyze results and to be reusable through 
# our data science process

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import seaborn as sns
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

