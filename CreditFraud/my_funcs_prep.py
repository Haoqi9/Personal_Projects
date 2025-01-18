"""
@author: Hao Qi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve

###############################################################################################################################

def plot_cat_barh(
    df: pd.DataFrame,
    cat_x: str,
    color_map=None,
    text_auto=True,
    template='plotly_white',
    opacity=0.9,
    title=None,
    **kwargs
):
    # Get df with count and relative freq in pct for cat variable.
    df_count = pd.Series(df[cat_x].value_counts(dropna=False), name='count')
    df_pct = pd.Series(df[cat_x].value_counts(dropna=False, normalize=True).round(3), name='freq_pct')
    df_pct = np.round(df_pct * 100, 2)
    df_freq = pd.concat([df_count, df_pct], axis=1).reset_index()

    if color_map is None:
        color_map = px.colors.qualitative.D3
    
    # Plot horizontal bar plot with count and freq in pct info.
    fig = px.bar(
        data_frame=df_freq,
        y=cat_x,
        x='freq_pct',
        color=cat_x,
        hover_name=(cat_x),
        hover_data={
            cat_x:False,
            'freq_pct':True,
            'count':True
        },
        color_discrete_sequence=color_map,
        opacity=opacity,
        text_auto=text_auto,
        template=template,
        **kwargs
    )
    
    if title is None:
        title = f"{cat_x}"

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,  # Centered title
            'xanchor': 'center',  # Anchor point for x coordinate
            'yanchor': 'top',  # Anchor point for y coordinate
            'font': {'size': 22, 'color': 'black'}  # Title font properties
        },
        yaxis=dict(title='Category'),
        xaxis=dict(title='Relative frequency (%)')
    )

    return fig

###############################################################################################################################

def plot_freq_barh_byCat(
    y_ser: pd.Series,
    x_ser: pd.Series,
    color_map=None,
    text_auto=True,
    template='plotly_white',
    opacity=0.9,
    **kwargs    
):
    if color_map is None:
        color_map = px.colors.qualitative.D3

    x, y = x_ser.name, y_ser.name
    
    # Get df with freq pct and count.
    # Get long format for both: cat with categories and values with frequencies.
    df_pct = pd.crosstab(index=x_ser, columns=y_ser, normalize='index', dropna=False).reset_index()
    freq_cols = df_pct.iloc[:, 1:].columns.tolist()
    df_pct[freq_cols] = (df_pct[freq_cols] * 100).round(2)
    df_pct_melt = df_pct.melt(id_vars=[x], value_name='freq_pct')
    df_count = pd.crosstab(index=x_ser, columns=y_ser, dropna=False).reset_index() 
    df_count_melt = df_count.melt(id_vars=[x], value_name='count')
    df_freq = pd.concat([df_pct_melt, df_count_melt['count']], axis=1)

    fig = px.bar(
        df_freq,
        y=x,
        x='freq_pct',
        orientation='h',
        color=y,
        hover_data={'count':True, x:False, y:False},
        hover_name=y,
        text_auto=text_auto,
        color_discrete_sequence=color_map,
        opacity=opacity,
        template=template,
        **kwargs
    )
    
    # Set title and axis labels.
    fig.update_layout(
        title={
            'text': f"Distribution of <b>{y}</b> by <b>{x}</b>",
            'x': 0.5,  # Centered title
            'xanchor': 'center',  # Anchor point for x coordinate
            'yanchor': 'top',  # Anchor point for y coordinate
            'font': {'size': 20, 'color': 'black'}  # Title font properties
        },
        yaxis=dict(title='Category'),
        xaxis=dict(title='Relative frequency (%)')
    )

    return fig

###############################################################################################################################

def plot_num_hist_box(
    df: pd.DataFrame,
    num_x: str,
    color_map=None,
    template='plotly_white',
    opacity=0.9,
    title=None,
    **kwargs    
):
    if color_map is None:
        color_map = px.colors.qualitative.D3
        
    fig = px.histogram(
        data_frame=df,
        x=num_x,
        marginal='box',
        opacity=opacity,
        template=template
    ) 

    # Add vertical mean line indicator.
    mean = df[num_x].mean()
    fig.add_vline(
        x=mean,
        line=dict(
            color='red',
            width=2,
            dash='dash',
        ),
        opacity=0.4,
        annotation=dict(
            text=f"(mean: {mean:,.2f})",
            font=dict(size=9.5, color='red')
        )
    )

    if title is None:
        title = num_x

    # Set title and axis labels.
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,  # Centered title
            'xanchor': 'center',  # Anchor point for x coordinate
            'yanchor': 'top',  # Anchor point for y coordinate
            'font': {'size': 22, 'color': 'black'}  # Title font properties
        },
        yaxis=dict(title='Count'),
        xaxis=dict(title='Bin')
    )

    return fig

###############################################################################################################################

def plot_hist_box_byCat(
    df: pd.DataFrame,
    num: str,
    cat: str,
    show_mean_vline=True,
    color_map=None,
    template='plotly_white',
    opacity=0.9,
    **kwargs    
):
    if color_map is None:
        color_map = px.colors.qualitative.D3 

    fig = px.histogram(
        df,
        x=num,
        color=cat,
        marginal='box',
        color_discrete_sequence=color_map,
        opacity=opacity,
        template=template,
        **kwargs
    )
    
    df_count = df[cat].value_counts(dropna=True).reset_index()
    for i, row in df_count.iterrows():
        # Add vertical mean indicator line for each category. 
        category_mask = df[cat] == row[cat]
        mean = df[category_mask][num].mean()
    
        if show_mean_vline is True:
            fig.add_vline(
                x=mean,
                line=dict(
                    color=color_map[i],
                    width=2,
                    dash='dash',
                ),
                opacity=0.45,
            )
    
        # Add mean value in legend for each category.
        if i > 0:
            # fig.data tuple contains hist followed by boxplot, so 0, 2, 4, 6, ...
            i = i * 2
        fig.data[i].name += f" ({mean:.2f}, {row['count']})"
    
    # Modify legend title.
    fig.update_layout(
        legend_title_text=f"{cat} (mean, count)",
        legend=dict(
            title=dict(font=dict(size=14), side="top")
        )
    )

    # Set title.
    fig.update_layout(
        title={
            'text': f"Distribution of <b>{num}</b> by <b>{cat}</b>",
            'x': 0.5,  # Centered title
            'xanchor': 'center',  # Anchor point for x coordinate
            'yanchor': 'top',  # Anchor point for y coordinate
            'font': {'size': 20, 'color': 'black'}  # Title font properties
        }
    )
    return fig

###############################################################################################################################

def plot_box_byCat(
    df: pd.DataFrame,
    num: str,
    cat: str,
    show_mean_hline=True,
    color_map=None,
    template='plotly_white',
    **kwargs 
):
    if color_map is None:
        color_map = px.colors.qualitative.D3 
        
    fig = px.box(
        df,
        x=cat,
        y=num,
        color=cat,
        color_discrete_sequence=color_map,
        template=template,
        **kwargs
    )
    
    df_count = df[cat].value_counts(dropna=True).reset_index()
    for i, row in df_count.iterrows():
        # Add vertical mean indicator line for each category. 
        category_mask = df[cat] == row[cat]
        mean = df[category_mask][num].mean()
    
        if show_mean_hline is True:
            fig.add_hline(
                y=mean,
                line=dict(
                    color=color_map[i],
                    width=2,
                    dash='dash',
                ),
                opacity=0.45,
            )
    
        # Add mean value in legend for each category.
        fig.data[i].name += f" ({mean:.2f}, {row['count']})"
    
    # Modify legend title.
    fig.update_layout(
        legend_title_text=f"{cat} (mean, count)",
        legend=dict(
            title=dict(font=dict(size=14), side="top")
        )
    )
    
    # Set title.
    fig.update_layout(
        title={
            'text': f"Distribution of <b>{num}</b> by <b>{cat}</b>",
            'x': 0.5,  # Centered title
            'xanchor': 'center',  # Anchor point for x coordinate
            'yanchor': 'top',  # Anchor point for y coordinate
            'font': {'size': 20, 'color': 'black'}  # Title font properties
        }
    )

    return fig

###############################################################################################################################

def group_infreq_labels(
    cat_series: pd.Series,
    threshold,
    label,
) -> pd.Series:
    # Create a copy.
    cat_series = cat_series.copy()

    # Get frequencies for each label.
    cat_freq = cat_series.value_counts(normalize=True)
    
    # Get a list of infrequent labels in cat variable.
    infreq_labels = [cat_freq.index[i] for i, freq in enumerate(cat_freq) if freq <= threshold]
    
    # Group infrequent labels.
    cat_series = pd.Series(np.where(cat_series.isin(infreq_labels), label, cat_series), name=cat_series.name)
    
    return cat_series

###############################################################################################################################

def binary_pred_kdeplots(X_test,
                         y_test,
                         predictor,
                         ax,
                         prob_thresh=0.5,
                         youden_cutoff=False,
                         ) -> plt.Axes:
    """
    Returns:
    ---
    - sns.kdeplot: Matplotlib Axes with the KDE plot.
    - Plot Kernel Density Estimation (KDE) plots for binary predictions based on a specified threshold.

    Parameters:
    ---
    - `X_test`: array-like or pd.DataFrame, features for testing.
    - `y_test`: array-like, true binary labels.
    - `predictor`: sklearn classifier or any model with a 'predict_proba' method.
    - `ax`: Matplotlib Axes, the axis on which to plot the KDE.
    - `prob_thresh`: float, probability threshold for binary classification (default is 0.5).
    - `youden_cutoff`: bool, whether to use the Youden Index to determine the 
    optimal threshold that maximizes both specificity(0) and sensitivity(1) (default is False).
    """
    y_pred_prob = predictor.predict_proba(X_test)[:,1]
    cutoff_name = 'cutoff'
    
    def cutoff_youden(y_test, y_pred_prob):
        """
        Calculate the cutoff threshold based on the Youden Index.
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        j_scores = tpr + (1 - fpr) - 1
        j_ordered = sorted(zip(j_scores, thresholds))
        return j_ordered[-1][1]

    if youden_cutoff is True:
        optimal_cutoff = cutoff_youden(y_test, y_pred_prob)
        prob_thresh = optimal_cutoff
        cutoff_name = 'Youden cutoff'

    ax.axvline(prob_thresh, color='red', alpha=0.3, label=f'cutoff = {prob_thresh}')
    ax.set_title(f'{cutoff_name} = {prob_thresh:.3f}', fontsize=10)
    return sns.kdeplot(x=y_pred_prob, hue=y_test, fill=True, alpha=0.4, ax=ax)

###############################################################################################################################

def feature_importance_plot(tree_predictor,
                            n_rows:int=None,
                            return_df=False,
                            bottom=False,
                            figsize=(8,6),
                            palette='tab10'
                            ) -> plt.Axes | pd.DataFrame:
    """
    Returns:
    --
    - Plot feature importance based on a tree-based predictor.

    Parameters:
    --
    - ``tree_predictor (object)``: A tree-based predictor (e.g., DecisionTreeClassifier, RandomForestRegressor).
    - ``bottom (bool, optional)``: If True, plot the bottom features; if False, plot the top features. Default is False.
    - ``n_rows (int, optional)``: Number of features to include in the plot. If None, all features are included. Default is None.
    - ``figsize (tuple, optional)``: Figure size. Default is (8, 6).
    - ``palette (str or list, optional)``: Color palette for the barplot. Default is 'viridis'.
    - ``return_df (bool, optional)``: Returns the df with feature importance measure (%) and feature names (index). Default is False.
    A subset df of the top or bottom features is possible.

    """
    # Bottom or top features
    if bottom is False:
        sort_type = 'Top'
    else:
        sort_type = 'Bottom'
    
    if tree_predictor.__class__.__name__ not in ["LGBMClassifier", "LGBMRegressor"]:
        df_feature = pd.DataFrame({'feature_importance': tree_predictor.feature_importances_},
                              index=tree_predictor.feature_names_in_)
    else:
        df_feature = pd.DataFrame({'feature_importance': tree_predictor.feature_importances_},
                            index=tree_predictor.feature_name_)
    
    # number of rows/features
    if n_rows:
        n = n_rows
    else: 
        n = df_feature.shape[0]
        
    sub_feature = df_feature.sort_values('feature_importance', ascending=bottom)[:n]
    
    if return_df:
        return sub_feature
    
    plt.figure(figsize=figsize)
    sns.barplot(y=sub_feature.index, x=sub_feature['feature_importance'], palette=palette, hue=sub_feature.index, legend=False)

    for idx, measure in enumerate(sub_feature['feature_importance']):
        plt.annotate(xy=(measure, idx), text=f'{measure:.2f}', fontsize='x-small')

    _, x_right = plt.xlim()
    plt.xlim(right=x_right + 1)
    plt.title(f'Feature importance ({sort_type} {n} out of {df_feature.shape[0]}) | {tree_predictor.__class__.__name__}', fontsize='large')
    
    try:
        criterion = tree_predictor.criterion
    except:
        criterion = 'UNKOWN'
    
    plt.ylabel('Features')
    plt.xlabel(f'Total amount {criterion} decreased by splits', fontsize='small')
    plt.show()