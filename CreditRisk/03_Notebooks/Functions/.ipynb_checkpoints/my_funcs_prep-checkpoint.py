"""
@author: Hao Qi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from typing import Literal
from math import ceil
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import (LearningCurveDisplay,
                                     cross_val_score,
                                     RepeatedKFold
                                     )
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             RocCurveDisplay,
                             auc,
                             precision_recall_curve,
                             PrecisionRecallDisplay,
                             )

###############################################################################################################################

def describe_custom(df,
                    decimals=2,
                    sorted_nunique=True
                    ) -> pd.DataFrame:
    """
    Generate a custom summary statistics DataFrame for the input DataFrame.

    Parameters:
    ---
    - `df (pd.DataFrame)`: Input DataFrame for which summary statistics are calculated.
    - `decimals (int, optional)`: Number of decimal places to round the results to (default is 2).
    - `sorted_nunique (bool, optional)`: If True, sort the result DataFrame based on the 'nunique' column
      in descending order; if False, return the DataFrame without sorting (default is True).

    Returns:
    ---
    pd.DataFrame: A summary statistics DataFrame with counts, unique counts, minimum, 25th percentile,
    median (50th percentile), mean, standard deviation, coefficient of variation, 75th percentile, and maximum.
    """
    def q1_25(ser):
        return ser.quantile(0.25)

    def q2_50(ser):
        return ser.quantile(0.50)

    def q3_75(ser):
        return ser.quantile(0.75)
    
    def CV(ser):
        return ser.std()/ser.mean()

    df = df.agg(['count','nunique', 'mean', 'std', CV, q1_25, q2_50, q3_75, 'min', 'max']).round(decimals).T    
    if sorted_nunique is False:
        return df
    else:
        return df.sort_values('nunique', ascending=False)

###############################################################################################################################

def histogram_boxplot(series,
                      figsize=(8,6),
                      show_qqplot=False,
                      extra_title=None,
                      IQR_multiplier=3,
                      show_IQR=False,
                      show_maxmin=True,
                      kde=True,
                      **kwargs
                      ) -> plt.Axes:
    """
    Returns:
    ---
    Generate a combined histogram and boxplot for a given series.

    Parameters:
    ---
    - ``series (pd.Series)``: The input series for analysis.
    - ``figsize (tuple, optional)``: Size of the figure (width, height). Default is (7, 6).
    - ``extra_title (str, optional)``: Additional title text for the plot. Default is None.
    - ``IQR_multiplier (float, optional)``: Multiplier for determining the whiskers in the boxplot. Default is 1.5.
    - ``show_IQR (bool, optional)``: Whether to display additional information about the interquartile range (IQR). Default is False.
    - ``kde (bool, optional)``: Whether to plot the kernel density estimate along with the histogram. Default is True.
    - ``**kwargs``: Additional keyword arguments to be passed to sns.histplot.

    Notes:
    ---
    - The function generates a combined histogram and boxplot for visualizing the distribution of the input series.
    - Additional statistical information, including quartiles, mean, skewness, kurtosis, and normality test results, is displayed.
    - Whiskers in the boxplot are determined by multiplying the interquartile range (IQR) by the specified IQR_multiplier.
    ```
    """
    # Get n + missing
    n = series.shape[0] - series.isna().sum()
    total_n = series.shape[0]
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=series, whis=IQR_multiplier, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=series, ax=ax_hist2, kde=kde, **kwargs)
    
    mean = np.mean(series)
    median = series.quantile(.5)
    p_25 = series.quantile(.25)
    p_75 = series.quantile(.75)
    IQR = IQR_multiplier * (p_75 - p_25)
    std = series.std()
    min = np.min(series)
    max = np.max(series)
    
    if show_maxmin is True:
        ax_hist2.axvline(min, color='grey', lw=1.5, alpha=0.15, label= 'Min: ' + f'{min:.2f}')
    if show_IQR is True:
        ax_hist2.axvline(p_25 - IQR, color='orange', lw=1.2, alpha=0.25, label= f'Q1 - {IQR_multiplier}*IQR: ' + f'{p_25 - IQR:.2f}')
    ax_hist2.axvline(p_25,color='black', lw=1.2, linestyle='--', alpha=0.35, label='Q1: ' + f'{p_25:.2f}')
    ax_hist2.axvline(mean,color='r', lw=2.5, alpha=0.45, label= 'Mean: ' + f'{mean:.2f}')
    ax_hist2.axvline(median,color='black', lw=1.5, linestyle='--', alpha=0.6, label='Q2: ' + f'{median:.2f}')
    ax_hist2.axvline(p_75,color='black', lw=1.2, linestyle='--', alpha=0.35, label='Q3: ' + f'{p_75:.2f}')
    if show_IQR is True:
        ax_hist2.axvline(p_75 + IQR, color='orange', lw=1.2, alpha=0.25, label= f'Q3 + {IQR_multiplier}*IQR: ' + f'{p_75 + IQR:.2f}')
        ax_hist2.axvspan(p_25 - IQR, p_75 + IQR, facecolor='yellow', alpha=0.15)
    if show_maxmin is True:
        ax_hist2.axvline(max, color='grey', lw=1.5, alpha=0.15, label= 'Max: ' + f'{max:.2f}')
      
    suptitle_text = f'"{series.name}"'
    if extra_title:
        suptitle_text += f" | {extra_title}"
        
    ax_box2.set_title(f'n: {n}/{total_n} | n_unique: {series.nunique():.0f} | std: {std:.2f} | skew: {series.skew():.2f} | kurt: {series.kurt():.2f}',
                fontsize=10)
    ax_hist2.set_xlabel(None)
    ax_box2.set_xlabel(None)
    # Mostrar gráfico
    f2.suptitle(suptitle_text, fontsize='xx-large')
    plt.legend(bbox_to_anchor=(1,1), fontsize='small')
    
    if show_qqplot is True:
        sm.qqplot(series.dropna(), fit=True, line='45', alpha=0.25)
        
###############################################################################################################################

def barh_plot(series,
              sort=True,
              extra_title=None,
              figsize=(7,6),
              xlim_expansion=1.15,
              palette='tab10',
              **kwargs
              ) -> plt.Axes:
    """
    Returns:
    ---
    - Create a horizontal bar plot for a categorical series.

    Parameters:
    ---
    - ``series (pandas.Series)``: The categorical data to be plotted.
    - ``sort (bool, optional)``: Whether to sort the bars by count. Default is True.
    - ``xlim_expansion (float, optional)``: Factor to expand the x-axis limit. Default is 1.15.
    - ``**kwargs``: Additional keyword arguments to pass to seaborn's countplot function.

    Notes:
    ---
    - The function creates a horizontal bar plot for the specified categorical series.
    - The bars can be sorted by count if 'sort' is True.
    - The function also annotates the bars with count and proportion information.
    - The x-axis limit is expanded by a factor of 'xlim_expansion'.

    Example:
    ---
    ```python
    # Sample data
    data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A'])

    # Create a horizontal bar plot
    barh_plot(data, sort=True, xlim_expansion=1.1, palette='viridis')
    ```
    """
    plt.figure(figsize=figsize)
    
    sns.countplot(y=series,
                width=0.5,
                order=series.value_counts(sort=sort).index,
                palette=palette,
                **kwargs
                )
    
    counts_no_order = series.value_counts(sort=sort)
    props_no_order = series.value_counts(sort=sort, normalize=True)
        
    for i, (count, prop) in enumerate(zip(counts_no_order, props_no_order)):
        plt.annotate(f' ({count}, {prop:.0%})', (count, i), fontsize=8)

    suptitle_text = f"'{series.name}'"
    if extra_title:
        suptitle_text += f" | {extra_title}"
    
    plt.ylabel('')
    plt.suptitle(suptitle_text, fontsize='xx-large')
    plt.title(f"n = {series.count()}/{series.size} | n_unique = {series.nunique()} | sort = {sort}")
    # Set xlimit
    _, xlim_r = plt.xlim()
    plt.xlim(right=xlim_r*xlim_expansion)

###############################################################################################################################

def cat_num_plots(
    data: pd.DataFrame,
    y: str,
    x: str,
    plot_type: Literal['box', 'violin']='box',
    log_yscale=False,
    n_adj_param=0.1,
    n_size=6.5,
    bar_alpha=0.1,
    extra_title=None,
    palette='tab10',
    **kwargs
):

    fig, axes = plt.subplots(figsize=(8,6))

    if plot_type == 'violin':
        ax = sns.violinplot(data=data, y=y, x=x, hue=x, legend=False, palette=palette, ax=axes, **kwargs)
    elif plot_type == 'box':
        ax = sns.boxplot(data=data, y=y, x=x, hue=x, legend=False, palette=palette, ax=axes, **kwargs)
    else:
        raise Exception("Must choose between plot_type: ['violin', 'box']!")

    if log_yscale is True:
        ax.set_yscale('log')
    
    if pd.api.types.is_categorical_dtype(data[x]):
        order = None
    else:
        order = data[x].unique()
    
    if data[x].dtype.name == 'object':
        sort = False
    elif data[x].dtype.name == 'category':
        sort = True
    else:
        raise TypeError(f"{x} variable is neither of object nor category dtype. It is {data[x].dtype.name}!")
    
    sns.countplot(data=data, x=x, ax=ax.twinx(), color='gray', order=order, alpha=bar_alpha)
    for i, (category_v, group) in enumerate(data.groupby(x, sort=sort)):
        n = len(group)
        plt.annotate(n, (i+n_adj_param, n), fontsize=n_size, color='blue')

    suptitle_text = f"{data[y].name} by {data[x].name}"
    if extra_title:
        suptitle_text += f" | {extra_title}"
      
    plt.suptitle(suptitle_text, fontsize=15, fontweight='bold')
    plt.title(f"n = {data[x].count()}/{data[x].size} | n_unique = {data[x].nunique()}", fontsize=10)
    axes.tick_params(axis='x', rotation=45, labelsize=9.5, labelrotation=45)

    return fig

###############################################################################################################################

def kdeplot_by_class(
    df: pd.DataFrame,
    x_num: str,
    y_cat: str,
    figsize=(8,6)
):
    """
    Plot kernel density estimation (KDE) plot grouped by a categorical variable.

    Parameters:
    ---
    - df (pd.DataFrame): The pandas DataFrame containing the data.
    - x_num (str): The name of the numerical column to be plotted on the x-axis.
    - y_cat (str): The name of the categorical column to be used for grouping.
    - figsize (tuple, optional): The size of the figure (width, height). Defaults to (8, 6).

    Returns:
    ---
    - fig (plt.Figure): The resulting matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(
        data=df,
        x=x_num,
        hue=y_cat,
        fill=True,
        palette='tab10',
        ax=ax
    )

    temp = df.groupby(y_cat, observed=False)[x_num].median()
    cats_indexes = temp.index
    cats_medians = temp.values

    y_axis = plt.ylim()[1] / 2
    position_decrease = 1
    for index, median in zip(cats_indexes, cats_medians):
        ax.axvline(median, ls='--', alpha=0.2, color='blue')
        plt.annotate(
            text=f"{index}: {median}",
            xy=(median, y_axis / position_decrease),
            fontsize=6,
            color='blue',
        )
        
        position_decrease += 0.8

    ax.set_ylabel('')
    ax.set_xlabel('')
    fig.suptitle(x_num, fontweight='bold')

    return fig

###############################################################################################################################

def class_balance_barhplot(x,
                           y,
                           text_size=9,
                           figsize=(8,6)
                           ):
    """
    Plot class balance bar horizontal plot.

    Parameters:
    -----------
    x : pandas.Series
        Input feature.
    y : pandas.Series
        Target variable.
    text_size : int, optional
        Font size for annotation text (default is 9).
    figsize : tuple, optional
        Figure size (width, height) in inches (default is (8, 6)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df_pct = pd.crosstab(x, y, normalize='index').sort_index(ascending=False)
    df_count = pd.crosstab(x, y).sort_index(ascending=False)
    
    # Binary or multiclass classification:
    n_y_classes = y.nunique()
    if n_y_classes > 2:
        df_pct.plot.barh(stacked=True, alpha=0.7, ax=ax)
        
        for i in range(0, len(df_pct.index)):
            pct_list = [str(np.round(pct, 2)) for pct in df_pct.iloc[i,:]]
            ax.annotate(
                text='p: ' + ' / '.join(pct_list),
                xy=(0.1, i + 0.1),
                alpha=0.8,
                color='blue'
            )
            
        for i in range(0, len(df_count.index)):
            pct_list = [str(np.round(pct, 2)) for pct in df_count.iloc[i,:]]
            ax.annotate(
                text='n: ' + ' / '.join(pct_list),
                xy=(0.1, i - 0.1),
                alpha=0.8,
                color='blue'
            )
        
    elif n_y_classes == 2:
        df_pct.plot.barh(stacked=True, color=['red', 'green'], alpha=0.7, ax=ax)

        for i, category in enumerate(df_pct.index):
            pct_0 = df_pct.iloc[:,0][category]
            pct_1 = df_pct.iloc[:,1][category]
            ax.annotate(text=f"{pct_0:.2f} | n={df_count.iloc[:,0][category]}",
                        xy=(0 + 0.01, i),
                        fontsize=text_size,
                        alpha=0.8,
                        color='blue'
                        )
            ax.annotate(text=f"{pct_1:.2f} | n={df_count.iloc[:,1][category]}",
                        xy=(0.92 - 0.1, i),
                        fontsize=text_size,
                        alpha=0.8,
                        color='blue'
                        )
            
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(f"Class distribution of '{y.name}' for categories in '{x.name}'")
    fig.suptitle(x.name, fontsize=15, fontweight='bold')
    
    return fig

###############################################################################################################################

def group_infreq_labels(
    cat_series: pd.Series,
    threshold=0.05,
    label='Rare',
) -> pd.Series:
    """
    Group infrequent labels in a categorical series.

    Parameters:
    ---
    - `cat_series` (pd.Series): The categorical series to process.
    - `threshold` (float, optional): The threshold below which labels are considered infrequent. Defaults to 0.05.
    - `label` (str, optional): The label to assign to infrequent values. Defaults to 'Rare'.

    Returns:
    ---
    pd.Series: A new categorical series with infrequent labels grouped under the specified 'label'.
    """
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

def prop_label_numx_plot(
    df: pd.DataFrame,
    y_cat: str,
    y_label: str,
    x_num: str,
    show_counts=False,
    first_n_rows: int=None,
    figsize=(12,6),
):
    """
    Plot the distribution of a categorical variable (y_label) within each numerical category (x_num).

    Parameters:
    ---
        df (pd.DataFrame): The DataFrame containing the data.
        y_cat (str): The name of the column containing the categorical variable.
        y_label (str): The label of the category to be analyzed within the categorical variable.
        x_num (str): The name of the column containing the numerical variable.
        show_counts (bool, optional): Whether to show counts on top of the bars. Defaults to False.
        first_n_rows (int, optional): Number of first rows to display. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12,6).

    Returns:
    ---
        matplotlib.axes._subplots.AxesSubplot: The matplotlib axes object.
    """
    if y_label not in df[y_cat].unique():
        raise Exception(f"y_label should be a string in {df[y_cat].unique()}")
        
    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if first_n_rows is None:
        counts_per_value = df[x_num].value_counts().sort_index()
        prop_label_by_numx = df[y_cat].groupby(df[x_num]).value_counts(normalize=True)[:, y_label]
    else:
        counts_per_value = df[x_num].value_counts().sort_index().iloc[:first_n_rows]
        prop_label_by_numx = df[y_cat].groupby(df[x_num]).value_counts(normalize=True)[:, y_label].iloc[:first_n_rows]
        
    ax = counts_per_value.plot.bar(ax=ax)
    ax.grid(alpha=0.15)

    if show_counts is True:
        for i, count in enumerate(counts_per_value):
            ax.annotate(
                text=count,
                xy=(i, count),
                fontsize=6
            )
            
    ax2 = ax.twinx()
    ax2.plot(prop_label_by_numx.index, prop_label_by_numx.values, color='green', alpha=0.6)
    return ax

###############################################################################################################################

def manage_outliers(series: pd.Series,
                       mode: Literal['check', 'return', 'winsor', 'miss']='check',
                       non_normal_crit: Literal['MAD', 'IQR']='MAD',
                       n_std=4,
                       multiplier=4,
                       MAD_threshold=8,
                       normal_cols: list=[],
                       alpha=0.05,
                       n_ljust=30
                       ) -> pd.Series:
   """
    Detect and manage outliers in a given numeric series.
    Often use the following way: ``df.apply(manage_outliers, mode='check')``

    Parameters
    ---
    - ``series (pd.Series)``: Input data series containing numeric values.
    - ``mode (str)``: Specifies the operation mode. Possible values: 'check' (default), 'return', 'winsor', 'miss'.
    Only essential for ``'return'`` mode!
    - ``n_std (float)``: Number of standard deviation away from the mean (normal distributions). Default is 4.
    - ``alpha (float)``: Significance level for normality tests (default=0.05).
    - ``multiplier (float)``: Multiplier for IQR-based outlier detection (default=4).
    - ``MAD_threshold (float)``: Threshold for Median Absolute Deviation (MAD)-based outlier detection (default=8).
    - ``normal_cols (list)``: List of cols assummed to be normal (default=[]).
    - ``n_ljust (int)``: Text alignment that displays determination method. (Default=30).
    
    Notes
    ---
    - For ``'check' mode``: pd.Series with lower, upper, and combined percentage of outliers (subset).
    - For ``'return' mode``: printed messages with a series of outlier values for each series/variable.
    Follow by using the ``function add_outliers_col()`` to get a new 'outlier col with list as values'.
    - For ``'winsor' mode``: Winsorized series based on lower and upper limits (df).
    - For ``'miss' mode``: Series with outliers replaced by NaN and information about missing values (series inplace change!).
   """
   # Check if mode is correct
   modes = ['check', 'winsor', 'return', 'miss']
   if mode not in modes:
       return f"Choose: {modes}"
   
   # Condición de asimetría y aplicación de criterio 1 según el caso
   if series.name == 'outlier_list':
       return series
   
   # Calcular primer cuartil     
   q1 = series.quantile(0.25)  
   # Calcular tercer cuartil  
   q3 = series.quantile(0.75)
   # Calculo de IQR
   IQR=q3-q1
   
   if series.name in normal_cols:
       message = f"'\033[1m{series.name}\033[0m':".ljust(n_ljust) + f"    normal (manual  | +-{n_std} std)"
       criterio1 = abs((series-series.mean())/series.std())>n_std
   else:
       n = series.shape[0] - series.isna().sum()
       if n < 50:
           method = 'shapiro'
           _, p_value = stats.shapiro(series)
       else:
           method = 'Kolmogo'
           _, p_value = stats.kstest(series, 'norm')
        
       if p_value >= alpha:
            message = f"'\033[1m{series.name}\033[0m':".ljust(n_ljust) + f"    normal ({method} | +-{n_std} std)"
            criterio1 = abs((series-series.mean())/series.std())>n_std
       else:
            if non_normal_crit == 'MAD':
                message = f"'\033[1m{series.name}\033[0m':".ljust(n_ljust) + f"non-normal ({method} | +-{MAD_threshold} MAD)"
                criterio1 = abs((series-series.median())/stats.median_abs_deviation(series.dropna()))>MAD_threshold
            elif non_normal_crit == 'IQR':
                message = f"'\033[1m{series.name}\033[0m':".ljust(n_ljust) + f"non-normal ({method} | +-{multiplier} IQR)"
                criterio1 = (series<(q1 - multiplier*IQR))|(series>(q3 + multiplier*IQR))
   
   lower = series[criterio1&(series<q1)].count()/series.dropna().count()
   upper = series[criterio1&(series>q3)].count()/series.dropna().count()
   
   # Salida según el tipo deseado
   if mode == 'check':
       print(message)
       ser = pd.Series({
           'lower (%)': np.round(lower*100, 2),
           'upper (%)': np.round(upper*100, 2),
           'All (%)': np.round((lower+upper)*100, 2)})
       return ser
   
   elif mode == 'return':
       print(f"\n---------------- \033[1m{series.name}\033[0m ----------------")
       print(series[criterio1].value_counts().sort_index())
       return None
   
   elif mode == 'winsor':
       winsored_series = series.clip(lower=series.quantile(lower, interpolation='lower'),
                                  upper=series.quantile(1 - upper, interpolation='higher'))
       return winsored_series
   
   elif mode == 'miss':
       missing_bef = series.isna().sum()
       series.loc[criterio1] = np.nan
       missing_aft = series.isna().sum()
      
   if missing_bef != missing_aft:
        print(series.name)
        print('Missing_bef: ' + str(missing_bef))
        print('Missing_aft: ' + str(missing_aft) +'\n')
        return (series)

###############################################################################################################################

def get_cramersV(x,
                 y,
                 n_bins=5,
                 return_scalar=False
                 ):
    """
    - Calculate Cramer's V statistic for the association between two categorical variables.
    - If either x or y is a continuous variable, the function discretizes it using fixed binning (default is 5 bins).

    Parameters:
    ---
    - `x (pd.Series)`: Predictor variable.
    - `y (pd.Series)`: Response variable.
    
    Notes:
    ---
    - Optimal binning is performed using the get_optbinned_x function if opt_binning is True.
    - The function then computes the Cramer's V statistic for the association between the two categorical variables using the contingency table.
    - The result is returned as a Pandas Series with the Cramer's V statistic and the variable name as the index.
    """
    # Discretizar x continua
    if pd.api.types.is_numeric_dtype(x) and (not (x.nunique() == 2)):
        x= pd.cut(x, bins=min(n_bins, x.nunique()))
            
    # Discretizar y continua
    if pd.api.types.is_numeric_dtype(y) and (not (y.nunique() == 2)):
        y = pd.cut(y, bins=min(n_bins, y.nunique()))
    
    name = f'CramersV: min(nunique, {n_bins}) bins'
        
    data = pd.crosstab(x, y).values
    vCramer = stats.contingency.association(data, method='cramer')
    
    if return_scalar is True:
        return vCramer
    else:
        return pd.Series({name:vCramer}, name=x.name)

###############################################################################################################################

def get_cramersV_matrix(df,
                        n_bins=5,
                        n_decimals=3
                        ) -> pd.DataFrame:
    """
    Returns:
    pd.DataFrame: A square matrix where each entry represents the Cramer's V value
                  between the corresponding pair of columns in the input DataFrame.

    Parameters:
    - `df (pd.DataFrame)`: The input DataFrame containing categorical columns.
    - `n_bins (int)`: Number of bins to use for discretizing continuous variables (default is 5).
    - `n_decimals (int)`: Number of decimals to round the resulting matrix values (default is 3).
    """
    # Initialize an empty DataFrame for Cramer's V matrix
    num_cols = len(df.columns)
    cramer_matrix = pd.DataFrame(np.zeros((num_cols, num_cols)), columns=df.columns, index=df.columns)
    
    # Iterate over each pair of columns and calculate Cramer's V
    for col1 in df.columns:
        for col2 in df.columns:
            cramers_v = get_cramersV(df[col1], df[col2],
                                     n_bins=n_bins,
                                     return_scalar=True
                                     )
            cramer_matrix.at[col1, col2] = cramers_v
    
    return cramer_matrix.round(n_decimals)

###############################################################################################################################

def association_barplot(df_widefmt: pd.DataFrame,
                        y: pd.Series=None,
                        abs_value=False,
                        extra_title=None,
                        xlim_expansion=1.15,
                        text_size=8,
                        text_right_size=0.0003,
                        palette='coolwarm',
                        figsize=(6,5),
                        title_size=14,
                        no_decimals=False,
                        ascending=False,
                        sort=True,
                        **kwargs
                        ) -> plt.Axes:
    """
    - Generate a barplot to visualize the association of predictors with a target variable.
    - Equivalent for Pearson corr df: 
    - `df_pearson = pd.DataFrame(X.select_dtypes(np.number).apply(lambda x: np.corrcoef(x, y)[0,1])).T`

    Parameters
    ---
    - `df_widefmt (pd.DataFrame)`: Wide-format DataFrame containing predictor variables.
    - `y (pd.Series)`: Target variable.
    - `palette (str or list of str, optional)`: Color palette for the barplot. Default is 'Reds'.
    - `extra_title (str, optional)`: Additional title text to be appended to the plot title. Default is None.
    - `figsize (tuple, optional)`: Figure size in inches. Default is (7, 6).
    - `xlim_expansion (float, optional)`: Expansion factor for the x-axis limit. Default is 1.15.
    - `text_size (int, optional)`: Font size for annotation text. Default is 8.
    - `text_right_size (float, optional)`: Adjustment for the horizontal position of annotation text. Default is 0.001.
    - `**kwargs`: Additional keyword arguments to be passed to seaborn.barplot.

    Notes
    ---
    - The function sorts the predictor variables based on their association with the target variable in descending order.
    - The barplot is annotated with the corresponding association metric values.
    - The title of the plot includes the number of predictors and the name of the target variable.
    - If extra_title is provided, it is appended to the plot title.
    - The x-axis limit is adjusted based on xlim_expansion.
    - The resulting plot is displayed using matplotlib.pyplot.show().
    """
    metric_col = df_widefmt.T.columns[0]
    
    if sort is True:
        df_longfmt = df_widefmt.T.sort_values(metric_col, ascending=ascending)
    else:
        df_longfmt = df_widefmt.T
    
    hue = None
    if abs_value is True:
        df_longfmt['Sign'] = df_longfmt[metric_col].apply(lambda row: 'Negative' if row < 0 else 'Positive')
        df_longfmt[metric_col] = abs(df_longfmt[metric_col])
        df_longfmt.sort_values(metric_col, ascending=False, inplace=True)
        hue = df_longfmt['Sign']
        
    plt.figure(figsize=figsize)
    
    sns.barplot(x=df_longfmt[metric_col], y=df_longfmt.index,
                hue=hue, hue_order=['Negative', 'Positive'],
                palette=palette, **kwargs)
    
    if no_decimals is True:
        for i, col in enumerate(df_longfmt.index):
            plt.annotate(text=f'{df_longfmt[metric_col][col]}',
                    xy=(df_longfmt[metric_col][col] + text_right_size, i),
                    fontsize=text_size)
    else:
        for i, col in enumerate(df_longfmt.index):
            plt.annotate(text=f'{df_longfmt[metric_col][col]:.3f}',
                        xy=(df_longfmt[metric_col][col] + text_right_size, i),
                        fontsize=text_size)
    
    if y is not None:
        title_text = f"{len(df_longfmt.index)} Predictors association wrt '{y.name}'"
    else:
        title_text = f"{len(df_longfmt.index)} Predictors"
        
    if extra_title:
        title_text += f" | {extra_title}"
    
    plt.ylabel('Predictors')
    plt.title(title_text, fontsize=title_size)
    _, xlim_r = plt.xlim()
    plt.xlim(right=xlim_r*xlim_expansion)
    plt.tight_layout()
    if abs_value is True:
        plt.legend(loc='lower right', fontsize=9)
    plt.show()

###############################################################################################################################

def scatteplots_wrt_y(
    data: pd.DataFrame,
    y_name: str,
    n_cols=3,
    figsize=(14,13),
    sharey=False,
    **kwargs
):
    """
    Create scatter plots of numerical columns in a DataFrame with respect to a specified y-variable.

    Parameters:
    ---
    - `data` : pd.DataFrame
        DataFrame containing the data.
    - `y_name` : str
        Name of the y-variable (column) to be plotted against.
    - `n_cols` : int, optional (default=3)
        Number of columns for subplots layout.
    - `figsize` : tuple, optional (default=(14, 13))
        Size of the figure (width, height) in inches.
    - `sharey` : bool, optional (default=False)
        Whether to share y-axis among subplots.
    - `**kwargs` : additional keyword arguments
        Additional keyword arguments to be passed to sns.regplot.
    """
    fig, axes = plt.subplots(
        nrows=ceil(data.shape[1] / n_cols),
        ncols=n_cols,
        sharey=sharey,
        figsize=figsize
    )
    
    axes = axes.flatten()

    numeric_cols = data.select_dtypes(np.number).drop(columns=y_name).columns

    # Get rid of extra axes:
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    for i, col in enumerate(numeric_cols):
        sns.regplot(
            data=data,
            y=y_name,
            x=col,
            scatter_kws={'alpha':0.2},
            line_kws={'color':'red', 'alpha':0.4},
            ax=axes[i],
            **kwargs
        )
        
        pearson_corr = np.corrcoef(data[col], data[y_name])[0,1]
        axes[i].set_title(f"{col} (ρ={pearson_corr:.2f})")
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')

    fig.suptitle(f"Scatter plots of X wrt y='{y_name}'", fontweight='bold')
    plt.tight_layout()
    plt.show()

###############################################################################################################################

def ic_media(datos: pd.Series,
             nivel_confianza=0.95
             ) -> str:
    """
    Devuelve un mensaje con el intervalo de confianza de la media de la ``muestra`` dada y al ``nivel de confianza`` indicado. \n
    El nivel de confianza por defecto es del 0.95 (95%). t-student statistic utilizado.
    
    Required packages: pandas (``pd``), numpy (``np``), scipy (``sp``).
    """

    t_statistic = np.abs(stats.t.ppf((1 - nivel_confianza) / 2, df=len(datos) - 1))
    media = datos.mean()
    error_estandar = datos.std() / np.sqrt(len(datos))
    
    margen_error = t_statistic * error_estandar
    limite_inferior = media - margen_error
    limite_superior = media + margen_error
    
    print(f"La media muestral es {media:.3f}")
    print(f"El intervalo de confianza de '{datos.name}' al {nivel_confianza*100:.0f}% de confianza:")
    print(f"[{limite_inferior:.3f}, {limite_superior:.3f}].")
    print(f"Este intervalo contiene las medias muestrales el {nivel_confianza*100:.0f}% de las veces.") 
    print(f"Por ello, hay un {nivel_confianza*100:.0f}% de confianza de que este intervalo contenga la media poblacional.")

###############################################################################################################################

# Función para calcular intervalos de confianzas de diferencias de medias con distintas varianzas:
def ic_diff_medias(datos1: pd.Series,
                   datos2: pd.Series, 
                   nivel_confianza=0.95
                   ) -> str:
    """
    Returns
    ---
    - str: Mensaje con el resultado del cálculo del intervalo de confianza.
    - Calcula el intervalo de confianza para la diferencia de medias entre dos muestras independientes.

    Parameters
    ---
    - `datos1 (pd.Series)`: Serie de datos de la primera muestra.
    - `datos2 (pd.Series)`: Serie de datos de la segunda muestra.
    - `nivel_confianza (float, optional`): Nivel de confianza deseado para el intervalo.
                                        Por defecto es 0.95.
    """
    t_statistic = np.abs(stats.t.ppf((1 - nivel_confianza)/2, df=len(datos1) + len(datos2) - 2))
    diff_medias = datos1.mean() - datos2.mean()
    error_estandar = np.sqrt(datos1.var()/len(datos1) + datos2.var()/len(datos2)) 
    
    margen_error = t_statistic * error_estandar
    limite_inferior = diff_medias - margen_error
    limite_superior = diff_medias + margen_error
    
    print(f"La diferencia de medias muestral es \033[1m{diff_medias:.3f}\033[0m\n")
    print(f"El intervalo de confianza de la diferencia de medias entre '{datos1.name}' y '{datos2.name}' al {nivel_confianza*100:.0f}% de confianza:")
    print(f"[\033[1m{limite_inferior:.3f}, {limite_superior:.3f}\033[0m].\n")
    print(f"Este intervalo contiene la diferencia de la medias muestrales el {nivel_confianza*100:.0f}% de las veces.") 
    print(f"Por ello, hay un \033[1m{nivel_confianza*100:.0f}\033[0m% de confianza de que este intervalo contenga la diferencia de medias poblacional.")
    
###############################################################################################################################

def ic_proportion(datos_total: pd.Series,
                  datos_exito: pd.Series,
                  nivel_confianza=0.95
                  ) -> str:
    """
    Devuelve un mensaje con el intervalo de confianza de la proporción de la 
    ``muestra`` dada y al ``nivel de confianza`` indicado. \n
    El nivel de confianza por defecto es del 0.95 (95%). z-statistic es utilizado.
    
    Required packages: pandas (``pd``), numpy (``np``), scipy (``sp``).
    """
    z_statistic = np.abs(stats.norm.ppf((1 - nivel_confianza)/2))
    num_total = len(datos_total)
    num_exitos = len(datos_exito)
    p_muestral = num_exitos / num_total
    des_tipica = np.sqrt((p_muestral*(1 - p_muestral))/num_total)

    limite_inferior = p_muestral - z_statistic * des_tipica
    limite_superior = p_muestral + z_statistic * des_tipica

    print(f"La proporción muestral es {p_muestral:.3f}")
    print(f"El intervalo de confianza de la proporción al {nivel_confianza*100:.0f}% de confianza:")
    print(f"[{limite_inferior:.3f}, {limite_superior:.3f}].")
    print(f"Este intervalo contiene las proporciones muestrales el {nivel_confianza*100:.0f}% de las veces.") 
    print(f"Por ello, hay un {nivel_confianza*100:.0f}% de confianza de que este intervalo contenga la muestra poblacional.")

###############################################################################################################################

def get_pareto(target_metric: pd.Series,
               output: str = 'table',
               unit_of_analysis: str = None
               ) -> pd.DataFrame | plt.Axes:
    """Returns a Pareto DataFrame (default) or Pareto graph (``output = 'graph'``) from the ``target metric`` (series).
    
    Note that if the unit of analysis (``index``) is at the individual level and you need it at the macro level,
    you would need to aggregate it using groupby.
    The column used for groupby will become the new index for the new aggregated DataFrame.
    
    Optional: provide a name for the ``unit_of_analysis`` (Asian countries, individuals, Companies in US, etc.) for the graph title.
    
    Required packages: pandas (``pd``), numpy (``np``), matplotlib.pyplot (``plt``), plotly.express (``px``).
    """
    
    # Sort the series in descending order and convert to df
    df_pareto = target_metric.sort_values(ascending=False).to_frame()
    
    # Create ranking variable in relative terms
    df_pareto['Ranking'] = np.arange(start=1, stop=len(df_pareto) + 1)
    df_pareto['Rank_pct'] = df_pareto.Ranking.transform(lambda rank: rank / df_pareto.shape[0] * 100).round(3)
    
    # Create cumulative percentage variable
    df_pareto['Cumulative'] = df_pareto[target_metric.name].cumsum()
    df_pareto['Cumul_pct'] = df_pareto.Cumulative.transform(lambda accum: accum / max(df_pareto.Cumulative) * 100).round(3)
    
    # Create individual contribution in pct variable
    df_pareto['Contr_pct'] = target_metric.sort_values(ascending=False).mul(100).div(max(df_pareto.Cumulative)).round(3)
    
    # Filter the df by the columns: Ranking_pct and Cumulative_pct (plus the index col)
    df_pareto = df_pareto[['Rank_pct', 'Contr_pct', 'Cumul_pct']]
    
    # Output graph
    plt.style.use('ggplot')
    if output == 'graph':
        
        # Create the line plots
        fig = px.line(df_pareto, x='Rank_pct', y='Cumul_pct', 
                    labels={'Rank_pct': 'Ranking (%)', 'Cumul_pct': 'Cumulative (%)'}, 
                    title=f"Pareto Chart: Cumulative {target_metric.name} by ranking of {unit_of_analysis}")
        
        # Add the second line plot (Pareto line) and change its color to red
        pareto_trace = px.line(df_pareto, x='Rank_pct', y='Rank_pct').data[0]
        pareto_trace.line.color = 'red'
        fig.add_trace(pareto_trace)
        
        # Show the interactive plot
        fig.show()
    else:
        return (df_pareto)

###############################################################################################################################

def compare_dfs(df1: pd.DataFrame,
                df2: pd.DataFrame
                ) -> str:
    """
    Returns:
    ---
    - str: A message describing the differences between the two DataFrames.
    - The function compares column-wise and checks for differences in values, data types, rounding, and unique values.
    - Differences are printed to the console.

    Parameters:
    ---
    - ``df1 (pd.DataFrame)``: The first DataFrame to be compared.
    - ``df2 (pd.DataFrame)``: The second DataFrame to be compared.
    - ``round_amt (int)``: Number of decimal places to consider when comparing floating-point values. Default is 3.
    """
    bad = False  # Flag to track if there are any differences.
    
    unique_cols1 = []
    common_cols = []
    for col in df1.columns:
        # If a col in df1 is not present in df2, continue with the nex col to prevent raising error.
        if col not in df2.columns:
            unique_cols1.append(col)
            continue
        
        common_cols.append(col)
        
        s1 = df1[col]
        s2 = df2[col]
        
        # Check if the Series (column) in df1 is equal to the Series in df2.
        if s1.equals(s2):
            continue
        
        bad = True
        
        # Check if the data types of the Series are different.
        if s1.dtype != s2.dtype:
            print()
            print(f"'\033[1m{col}\033[0m' dtype differ {s1.dtype}(df1) vs {s2.dtype}(df2)\n")
        
        # If any different unique values
        if len(s1.unique()) != len(s2.unique()):
            print(f"'\033[1m{col}\033[0m' has diff length in unique values [df1:{len(s1.unique())}, df1:{len(s2.unique())}] (capped to 15 values):")
            if pd.api.types.is_numeric_dtype(s1) & pd.api.types.is_numeric_dtype(s2):
                print(f"\tdf1 ({len(s1.unique())}): {s1.unique()[:15].tolist()}")
                print(f"\tdf2 ({len(s2.unique())}): {s2.unique()[:15].tolist()}")            
            else:
                print(f"\tdf1 ({len(s1.unique())}): {s1.unique()[:15].tolist()}")
                print(f"\tdf2 ({len(s2.unique())}): {s2.unique()[:15].tolist()}")
        else:
            if (s1.unique() != s2.unique()).any():
                df1_rows_unique = df1[col].unique()[s1.unique() != s2.unique()]
                df2_rows_unique = df2[col].unique()[s1.unique() != s2.unique()]

                print(f"'\033[1m{col}\033[0m' has same length [df1:{len(s1.unique())}, df1:{len(s2.unique())}], but different unique values: {len(df1_rows_unique)}.")
                print(f"\tUnique values capped to 20 (df1, df2):\n{[(r1, r2) for r1, r2 in list(zip(df1_rows_unique, df2_rows_unique))[:20]]}")
            
        # Print the differing values for non-float columns (execpt for nan).
        s1_null_count = s1.isnull().sum() 
        s2_null_count = s2.isnull().sum() 
        if s1_null_count > 0:
            print(f"df1 got \033[1m{s1_null_count} null values\033[0m!")
        if s2_null_count > 0:
            print(f"df2 got \033[1m{s2_null_count} null values\033[0m!")
        if len(df1[s1.ne(s2)][col].dropna()) | len(df2[s2.ne(s1)][col].dropna()) > 0:
            print(f"'\033[1m{col}\033[0m' values differ from each other at index and values:" + 
                "\n" + '\t----df1----' + "\n" +  f"{df1[s1.ne(s2)][col]}" + "\n" +
                '\t----df2----' + "\n" +  f"{df2[s2.ne(s1)][col]}"
                )
            print()
    
    # Print df1 and df2 shapes and columns.
    unique_cols2 = [col for col in df2.columns if (col not in unique_cols1) and (col not in common_cols)]
    
    print()
    print(f"df1 shape: {df1.shape}")
    print(f"df2 shape: {df2.shape}")
    # # Check for columns that exist in one DataFrame but not the other.
    # diff_cols = set(df1.columns) ^ set(df2.columns)
    # if diff_cols:
    #     print(f"Different columns {diff_cols}")
    print(f"{len(common_cols)} Common columns: {common_cols}")
    print(f"{len(unique_cols1)} Unique columns (df1): {unique_cols1}")
    print(f"{len(unique_cols2)} Unique columns (df2): {unique_cols2}")
    
    # If no differences were found, print that the DataFrames are the same.
    if not bad:
        print('All COMMON COLUMNS have same values (equal)!')

###############################################################################################################################

def get_feat_selection_df(
    X_dm: pd.DataFrame,
    y: pd.Series,
    estimator_pi,
    scoring_pi,
    objective: Literal['regression', 'classification']='classification',
    random_state=None
):
    """
    Perform feature selection using Mutual Information and Permutation Importance.

    Parameters:
    - X_dm (pd.DataFrame): The feature matrix.
    - y (pd.Series): The target vector.
    - estimator_pi : A fitted estimator for permutation importance, must have a `fit` method.
    - scoring_pi : Scorer function or a valid scorer name, used to compute permutation importance.
    - objective (Literal['regression', 'classification'], optional): Type of problem, either 'regression' or 'classification'. Default is 'classification'.
    - random_state (int or None, optional): Random seed for reproducibility. Default is None.

    Returns:
    - pd.DataFrame: DataFrame containing variables, Mutual Information scores, Permutation Importance scores, rankings, and average rankings.
    """
    
    # Mutual Information:
    print(f"mutual information {objective} running...")
    if objective == 'classification':
        mutual_selector = mutual_info_regression(X_dm, y, random_state=random_state)
    elif objective == 'regression':
        mutual_selector = mutual_info_classif(X_dm, y, random_state=random_state)
    print(f"mutual information {objective} ended! \n")
    
    df_mutual = pd.DataFrame({
        'variable': X_dm.columns,
        'MI_score': mutual_selector
    })

    df_mutual['MI_ranking'] = df_mutual['MI_score'].rank(ascending=False)

    # Permutation Importance:
    estimator_pi.fit(X_dm, y)
    
    print(f"permutation importance using {estimator_pi.__class__.__name__} running...")
    permutation_selector = permutation_importance(
        estimator=estimator_pi,
        X=X_dm,
        y=y,
        scoring=scoring_pi,
        n_repeats=5,
        n_jobs=-1,
        random_state=random_state
    )
    print(f"permutation importance using {estimator_pi.__class__.__name__} ended!")

    df_permutation = pd.DataFrame({
        'variable': X_dm.columns,
        'PI_score': permutation_selector.importances_mean
    })
    
    df_permutation['PI_ranking'] = df_permutation['PI_score'].rank(ascending=False)

    # Combined:
    df_select = pd.merge(
        left=df_mutual,
        right=df_permutation,
        how='inner',
        on='variable',
    )

    # Reorder columns
    df_select = df_select[['variable', 'MI_score', 'PI_score', 'MI_ranking', 'PI_ranking']]
    
    # Add avg ranking and sort by it
    df_select['avg_ranking'] = (df_select['MI_ranking'] + df_select['PI_ranking']) / 2
    df_select = df_select.sort_values('avg_ranking')

    return df_select

###############################################################################################################################

def get_styled_df_greater_than(df,
                            number,
                            abs_value=False
                            ) -> pd.DataFrame:
    """
    Returns:
    --
    - Apply styling to a DataFrame to highlight values greater than or equal to a specified number.
    - pd.DataFrame: A styled DataFrame with background colors based on the specified conditions.

    Parameters:
    --
    - ``df (pd.DataFrame):`` The input DataFrame to be styled.
    - ``number (float):`` The threshold value. Values greater than or equal to this number will be highlighted.
    - ``abs_value=False (bool, optional):`` row value regardless of the sign. Default is False.

    Example:
    --
    ```python
    # Create a sample DataFrame
    data = {'A': [10, 5, 8, 12, 15], 'B': [3, 7, 9, 2, 10], 'C': [6, 8, 12, 5, 3]}
    df = pd.DataFrame(data)

    # Apply styling to highlight values greater than or equal to 8
    styled_df = get_styled_df_greater_than(df, number=8)

    # Display the styled DataFrame
    styled_df
    ```
    """
    number = number
    def color(value, number):
        if abs_value is True:
            value = np.abs(value)
            
        if value >= number:
            return('background-color: lightgreen')
        
        elif pd.isna(value):
            return('background-color: black')
        
        else:
            return('color: red')
    
    return df.style.applymap(color, number=number)

###############################################################################################################################

def show_sklearn_EvaMetrics():
    print("""
    Sklearn API link: https://scikit-learn.org/stable/modules/model_evaluation.html
    
    |----------------|------------------------------------------------------------------------------------------------------------------------------------|
    |      TYPE      |                                                           SCORINGS                                                                 |
    |----------------|------------------------------------------------------------------------------------------------------------------------------------|
    |                | 'accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score',                                         |
    | Classification | 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'neg_log_loss', 'precision',                                            |
    |                | 'recall', 'jaccard', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'.                      |
    |----------------|------------------------------------------------------------------------------------------------------------------------------------|
    |                | 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',                                            |
    |   Regression   | 'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2', 'neg_mean_poisson_deviance',       |
    |                | 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error', 'd2_absolute_error_score', 'd2_pinball_score', 'd2_tweedie_score'.|
    |----------------|------------------------------------------------------------------------------------------------------------------------------------|
    |                | 'adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score', 'fowlkes_mallows_score',                                |
    |   Clustering   | 'homogeneity_score', 'mutual_info_score', 'normalized_mutual_info_score',                                                          |
    |                | 'rand_score', 'v_measure_score'.                                                                                                   |
    |----------------|------------------------------------------------------------------------------------------------------------------------------------|
    """
    )

###############################################################################################################################

def compare_models_bias_var(dmatrices: list,
                              y_train: pd.Series,
                              estimators: list,
                              names: list = None,
                              scoring: str = None,
                              show_n_features=False,
                              ascending: bool = True,
                              n_splits=5,
                              n_repeats=20,
                              random_state=1,
                              figsize=(8,6),
                              n_decimals=3,
                              **kwargs
                              ):
    """
    Returns
    ---
    - Perform cross-validation for multiple models and display the results (ranking and boxplot).
    - `figsize=(14,9)` when more than 4 design matrices (models) are provided!
    - None (prints model rankings and displays a boxplot).

    Parameters
    ---
    - `dmatrices `(list): List of design matrices for each model.
    - `y_train` (pd.Series): Target variable in training set.
    - `estimators` (list): List of machine learning models to compare.
    - `scoring` (str, optional): Scoring metric for cross-validation. Default is None.
    - `n_splits` (int, optional): Number of splits in cross-validation. Default is 5.
    - `n_repeats` (int, optional): Number of times cross-validation is repeated. Default is 20.
    - `random_state` (int, optional): Seed for random state to ensure reproducibility. Default is 1.
    - `names` (list, optional): List of names for models. Default is None, in which case numeric identifiers are used.
    - `ascending` (bool, optional): Order of sorting scores. Default is True (ascending order).
    - `figsize` (tuple, optional): Size of the figure for the plot. Default is (8, 6).

    Notes:
    ---
    - The default scoring metric is 'roc_auc' for binary classification and 'neg_root_mean_squared_error'
      for regression, unless a custom scoring metric is specified.
    - The function uses RepeatedKFold cross-validation with the provided number of splits and repeats.
    - `If only one estimator is provided`, the function compares different design matrices for that estimator.
    - `If multiple estimators are provided`, the function compares the same design matrix for each estimator.
    """
    # Establecemos esquema de validacion fijando random_state (reproducibilidad)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    
    # Obtenemos los resultados para cada particion tr-tst para cada matriz de diseno
    default = False
    if scoring is None:
        default = True
        # Default scoring for classification is roc_auc
        if isinstance(y_train.dtype, pd.CategoricalDtype) or (y_train.nunique() == 2):
            scoring = 'roc_auc'
            ascending = False
        # Default scoring for regression is RMSE
        else:
            scoring = 'neg_root_mean_squared_error'
            ascending = True
    
    identifiers = list(range(1,len(dmatrices)+1))
    if names:
        identifiers = names
        
    arrays_dict = {}
    scores_list = []
    
    if len(estimators) == 1:
        for id, dmatrix in zip(identifiers, dmatrices):
            array = abs(cross_val_score(estimators[0], dmatrix, y_train, scoring=scoring, cv=cv, n_jobs=-1))
            
            if show_n_features is True:
                id += f"_{dmatrix.shape[1]}"
            
            # store scores array
            arrays_dict[id] = array
            # store mean and std scores
            mean_score_col_name  = f"{scoring.strip('neg_')}"
            scores_list.append({
                'Model'             : id,
                mean_score_col_name : np.mean(array).round(n_decimals),
                'std_score'         : np.std(array).round(n_decimals)
            })
    else:
        for estimator in estimators:
            estimator_name = estimator.__class__.__name__
            array = abs(cross_val_score(estimator, dmatrices[0], y_train, scoring=scoring, cv=cv, n_jobs=-1))
            arrays_dict[estimator_name] = array
            mean_score_col_name  = f"{scoring.strip('neg_')}"
            scores_list.append({
                'Model'             : estimator_name,
                mean_score_col_name : np.mean(array).round(n_decimals),
                'std_score'         : np.std(array).round(n_decimals)
            })
        
    df_scores = pd.DataFrame(scores_list).sort_values(mean_score_col_name, ascending=ascending)
    print("******************** RANKING ********************")
    print("*************************************************")
    print(df_scores)
    print()
    
    if len(dmatrices) > 4:
        figsize=(14,9)
    
    plt.figure(figsize=figsize)
    df_arrays_long = pd.DataFrame(arrays_dict).melt(var_name='Models', value_name=mean_score_col_name)
    sns.boxplot(data=df_arrays_long, y=mean_score_col_name, x='Models', palette='tab10', **kwargs)
    total_splits = len(array)
    plt.suptitle(f"Models comparison (bias and variance) | total splits per model = {total_splits}")
    plt.title(f"y = '{y_train.name}' | default scoring = {default} | n_diff_estimators = {len(estimators)}")
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.tight_layout()

###############################################################################################################################

def learning_curve_plot(estimator,
                         X_train,
                         y_train,
                         train_sizes=np.linspace(0.1, 1.0, 10), 
                         cv=5,
                         scoring='accuracy',
                         figsize=(10,7),
                         n_jobs=-1,
                         y_log_scale=False,
                         **kwargs
                         ) -> plt.Axes:
    """
    Plot learning curves for a given estimator.

    Parameters
    -----------
    - `estimator` : object
        An instance of an estimator that implements the 'fit' and 'predict' methods.
    - `X_train` : array-like or matrix.
    - `y_train` : array-like.
    - `train_sizes` : array-like, shape (n_ticks,), optional
        Relative or absolute numbers of training examples that will be used to generate the learning curve.
        If the dtype is float, it is regarded as a fraction of the maximum size of the training set
        (that is determined by the selected validation method), i.e., it has to be within (0, 1].
        Otherwise, it is interpreted as absolute sizes of the training sets. Note that for classification
        the number of samples usually have to be big enough to contain at least one sample from each class.
    - `cv` : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    - `scoring` : str or callable, optional, default: 'accuracy'
        A string (see model evaluation documentation) or a scorer callable object / function with signature
        scorer(estimator, X, y).
    - `figsize` : tuple, optional, default: (10,7)
        Width, height in inches.
    - `n_jobs` : int, optional, default: -1
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    - `y_log_scale` : bool, optional, default: False
        Whether to use a logarithmic scale for the y-axis.
    - `**kwargs`
        Additional keyword arguments to be passed to the LearningCurveDisplay.from_estimator method.
    """
    _, ax = plt.subplots(figsize=figsize)
    
    if scoring.startswith('neg_'):
       negate_score = True
    else:
       negate_score = False 
        
    common_parameters = {
        "X":X_train,
        "y":y_train,
        "train_sizes":train_sizes,
        "cv":cv,
        "score_type":"both",
        "n_jobs":n_jobs,
        "line_kw":{"marker": "o"},
        "std_display_style":"fill_between",
        "negate_score":negate_score,
        "scoring":scoring,
        "score_name":str(scoring),
    }
    
    estimator_name = f'{estimator.__class__.__name__}'
    LearningCurveDisplay.from_estimator(estimator=estimator, **common_parameters, ax=ax, **kwargs)
    ax.set_title(f"Learning Curve for " + estimator_name)
    
    if y_log_scale is True:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
###############################################################################################################################

def reg_predictions_scatter(regressor_predictor,
                     y,
                     X_set,
                     y_set,
                     figsize=(8,6),
                     show_RMSE=False,
                     test=True
                     ) -> plt.Axes:
    """
    Returns:
    --
    - Generate a scatter plot comparing actual vs. predicted values from a regression model.
    - The function plots a scatter plot with actual vs. predicted values, color-coded
    based on overestimations and underestimations. It also includes a dashed line
    representing perfect predictions and displays the mean squared error (MSE) in the title.

    Parameters:
    --
    - ``regressor_predictor (object)``: A trained regression model with a `predict` method.
    - ``y (pd.Series)``: The target variable.
    - ``X_set (pd.DataFrame)``: Test/Training set features.
    - ``y_set (pd.Series)``: True target values for the test/training set.
    - ``figsize (tuple, optional)``: Figure size (width, height). Default is (8, 6).
    - ``show_RMSE (bool, optional)``: Show RMSE if True instead of MSE. Default is False.
    - ``test=True (bool, optional)``: Predicting y_train or y_test. Default is y_test.

    Example:
    --
    >>> reg_predictions_scatter(model, y_train, X_test, y_test)
    """
    n_test = X_set.shape[0]
    
    y_hat = regressor_predictor.predict(X_set)

    df_pred = pd.DataFrame({
        'y_test': y_set,
        'y_hat': y_hat,
         # Create boolean mask for under- and overestimations as hue!
        'overestimation': np.where(y_hat >= y_set, 'Overestimated', 'Underestimated')
    })

    plt.figure(figsize=figsize)

    sns.scatterplot(data=df_pred, 
                    x='y_test',
                    y='y_hat', 
                    hue='overestimation', 
                    palette={'Overestimated': 'green', 'Underestimated': 'red'},
                    alpha=0.45
                    )
    
    sns.lineplot(data=df_pred, x='y_test', y='y_test', c='grey', ls='--', label='Perfect prediction')
    
    mean_sq_error = np.mean((y_set - y_hat)**2)
    
    if show_RMSE:
        show_error_score = np.sqrt(mean_sq_error)
        metric_name = 'RMSE'
    else:
        show_error_score = mean_sq_error
        metric_name = 'MSE'
    
    if test is True:
        set_name = 'Test'
    else:
        set_name = 'Training'
    
    plt.title(f"Actual vs Predicted '{y.name}' ({set_name} set: {n_test}) | {metric_name} = {show_error_score:.2f}")
    plt.ylabel('Predictions')
    plt.xlabel('Actual values')
    plt.legend(fontsize='small')
    plt.show()
    
    print(f'Minimum {set_name} {metric_name} = {show_error_score}')
    
###############################################################################################################################

def classif_report_threshold(predictor, 
                             X_set,
                             y_set,
                             threshold=0.5,
                             test=True,
                             extra_title=None,
                             n_decimals=3
                             ) -> pd.DataFrame:
    """
    Returns
    ---
    - pd.DataFrame: A DataFrame containing the confusion matrix.
    - Generate and print a classification report for a binary classifier with a given threshold.

    Parameters
    ---
    - `predictor`: The fitted binary classifier model.
    - `X_set`: The feature matrix of the dataset. Train or test set.
    - `y_set`: The true labels of the dataset.
    - `threshold` (float, optional): The probability threshold for binary classification. Default is 0.5.
    - `test` (bool, optional): If True, indicates that the dataset is a test set; otherwise, it's a training set.
    - `extra_title` (str, optional): Additional text to be added to the model name in the report.
    - `n_decimals` (int, optional): Number of decimals to round the metric values.
    """
    if test is True:
        set_type = 'Test'
    else:
        set_type = 'Training'
        
    
    if predictor.__class__.__name__ == 'SVC':
        y_hat = predictor.predict(X_set)
    else:
        probs = predictor.predict_proba(X_set)[:,1]
        y_hat = np.where(probs >= threshold, 1, 0)
    
    accuracy = np.round(accuracy_score(y_set, y_hat), 2)
    
    precision_1 = np.round(precision_score(y_set, y_hat), 2)
    recall_1 = np.round(recall_score(y_set, y_hat), 2)
    f1score_1 = np.round(f1_score(y_set, y_hat), 2)
    
    precision_0 = np.round(precision_score(y_set, y_hat, pos_label=0), 2)
    recall_0 = np.round(recall_score(y_set, y_hat, pos_label=0), 2)
    f1score_0 = np.round(f1_score(y_set, y_hat, pos_label=0), 2)
    
    cm = confusion_matrix(y_set, y_hat)
    cm_df = pd.DataFrame(cm,
                         index=['Actual: 0','Actual: 1'],
                         columns=['Predicted: 0','Predicted: 1'])

    cm_df['Actual: n'] = cm_df.sum(axis=1)

    # Howt to add a new row (dict) in a df with assigned index!
    cm_df.loc['Predicted: n'] = {
        'Predicted: 0': cm_df['Predicted: 0'].sum(),
        'Predicted: 1': cm_df['Predicted: 1'].sum(),
        'Actual: n': cm_df['Actual: n'].sum()
    }
    
    df_scores = pd.DataFrame([
        {'precision':precision_0, 'recall':recall_0, 'f1-score':f1score_0, 'sample':cm[0,:].sum()},
        {'precision':precision_1, 'recall':recall_1, 'f1-score':f1score_1, 'sample':cm[1,:].sum()},
    ], index=[0, 1])
    
    # Add additional text to model
    model_text = f'{predictor.__class__.__name__} '
    if extra_title:
        model_text += f'({extra_title})'
    
    print(model_text.center(46))
    print("-----------------------------------------------")
    print(f'{set_type} set | prob_thresh = {threshold} | accuracy = {accuracy}')
    print("-----------------------------------------------")
    print(df_scores)
    
    return cm_df

###############################################################################################################################

def classif_aucs_plot(predictors: list,
                 X_test,
                 y_test,
                 add_text_list: list=None,
                 pos_label=1,
                 return_df=False,
                 lw=1.5,
                 figsize=(12,6),
                 **kwargs
                 ) -> plt.Axes:
    """
    Plot ROC and Precision-Recall curves for multiple classifiers.

    Parameters:
    ---
    - `predictors` (list): List of fitted classifier models to be evaluated.
    - `X_test`: Feature matrix of the test dataset.
    - `y_test`: True labels of the test dataset.
    - `add_text_list` (list, optional): List of additional text to be added to each classifier's name in the legend.
    - `pos_label` (int, optional): The positive class label. Default is 1.
    - `return_df` (bool, optional): If True, return a DataFrame containing AUC scores for each classifier.
    - `lw` (float, optional): Line width for the curves. Default is 1.5.
    - `figsize` (tuple, optional): Figure size.
    - `**kwargs`: Additional keyword arguments for matplotlib.pyplot.subplots.

    Returns:
    ---
    - plt.Axes: Matplotlib Axes containing the ROC and Precision-Recall curves.

    Example:
    ---
    >>> rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    >>> gb_model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    >>> classif_aucs_plot([rf_model, gb_model], X_test, y_test, add_text_list=['RF', 'GB'], return_df=True)
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize, **kwargs)
    
    # No skill curve for ROC
    ax[0].plot(np.linspace(0,1, 10),
            np.linspace(0,1,10),
            ls='--',
            color='grey',
            alpha=0.4,            
            label='No Skill  (AUC = 0.500)'
            )
    
    # No skill curve for PR
    ax[1].plot(np.linspace(0,1, 10),
            np.linspace(1,0,10),
            ls='--',
            color='grey',
            alpha=0.4,            
            label='No Skill  (AUC = 0.500)'
            )
    
    if add_text_list:
        sep_string = '_'
        add_estimator_text = add_text_list
    else:
        sep_string = ''
        add_estimator_text = [""] * len(predictors)
    
    datasets = []
    for predictor, text in zip(predictors, add_estimator_text):
        y_hat_probs = predictor.predict_proba(X_test)[:,1]
        # Get false positive and true positive rates
        fprs, tprs, _ = roc_curve(y_test, y_hat_probs)
        # Get precision and recall rates for class 1
        precs, recalls, _ = precision_recall_curve(y_test, y_hat_probs, pos_label=pos_label)
        
        # Get data for each estimator
        predictor_name = f'{predictor.__class__.__name__}{sep_string}{text}'
        roc_auc_score  = auc(fprs, tprs)
        pr_auc_score   = auc(recalls, precs)
        
        datasets.append({
            'Predictor' : predictor_name,
            'ROC_auc'   : roc_auc_score,
            'PR_auc'    : pr_auc_score
        })
        
        # Display cruve visualizations
        RocCurveDisplay(fpr=fprs,
                        tpr=tprs
                        ).plot(ax=ax[0],
                               lw=lw,
                               label=predictor_name +
                               f" | AUC = {roc_auc_score:.3f}"
                        )
                        
        PrecisionRecallDisplay(precs,
                        recalls,
                        pos_label=pos_label
                        ).plot(ax=ax[1],
                               lw=lw,
                               label=predictor_name +
                               f" | AUC = {pr_auc_score:.3f}"
                        )
                        
    ax[0].set_title('ROC curves', fontsize='x-large')
    ax[0].legend(fontsize=8, loc='lower right')
    ax[1].set_title('Precision-Recall curves', fontsize='x-large')
    ax[1].legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.show()
    
    # Whether return df
    if return_df is True:
        return pd.DataFrame(datasets).sort_values('ROC_auc', ascending=False)

###############################################################################################################################

def get_max_roi_threshold(
    predictor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    impact_list: list,
    return_type: Literal['scalar', 'graph']='graph',
    list_prob_thresholds: list=None,
    show_positive_only=True,
    figsize=(10,7)
):
    """
    Calculate the maximum return on investment (ROI) threshold based on economic impact values for a binary classifier.

    Parameters:
    ---
    - `predictor`: The trained binary classifier model.
    - `X_test` (pd.DataFrame): Features DataFrame for testing.
    - `y_test` (pd.DataFrame): Target DataFrame for testing.
    - `impact_list` (list): List with the economic impact/value for each of the 4 rates in a confusion matrix. They should be ordered as [impact_TNR, impact_FPR, impact_FNR, impact_TPR].
    - `return_type` (Literal['scalar', 'graph'], optional): Type of return value. 'scalar' to return a tuple of (threshold, expected value), 'graph' to plot expected value against probability thresholds. Default is 'graph'.
    - `list_prob_thresholds` (list, optional): List of probability thresholds to evaluate. If None, defaults to 100 equally spaced thresholds between 0 and 1. Default is None.
    - `show_positive_only` (bool, optional): Flag to show only positive expected values on the graph. Default is True.

    Returns:
    ---
    - If `return_type is 'scalar'`, returns a tuple (max_value_thresh, max_value) where max_value_thresh is the probability threshold that maximizes expected value, and max_value is the corresponding maximum expected value.
    - If `return_type is 'graph'`, plots the expected value against probability thresholds and visualizes the threshold that maximizes expected value.
    """
    def get_expected_value(conf_matrix, impact_list):
        TNR, FPR, FNR, TPR = conf_matrix.ravel()
        expect_value = (TNR * impact_list[0]) + (FPR * impact_list[1]) + (FNR * impact_list[2]) + (TPR * impact_list[3])
        return expect_value
    
    if len(impact_list) != 4:
        raise Exception(f"Length of provided impact_list = {len(impact_list)}. `impact_list` is a list with the economic impact/value for each of the 4 rates in a confusion matrix. They should be ordered in the folliwing way: \033[1m[impact_TNR, impact_FPR, impact_FNR, impact_TPR]\033[0m")
    
    # Predicted probabilities for each row by ML model.
    probs = predictor.predict_proba(X_test)[:,1]
    
    if list_prob_thresholds is None:
        prob_thresholds = np.arange(0, 1, 0.01)  # 100 prob cutoff points.
    else:
        prob_thresholds = list_prob_thresholds
    
    # List of expected value for each prob treshold:
    expected_values_list = []
    for threshold in prob_thresholds:
        y_pred = np.where(probs >= threshold, 1, 0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        expected_value = get_expected_value(conf_matrix, impact_list)
        expected_values_list.append(tuple([threshold, expected_value]))
    
    df_temp = pd.DataFrame(expected_values_list, columns=['prob_threshold', 'expected_value'])
    
    max_value_thresh = df_temp.iloc[df_temp['expected_value'].idxmax(), 0]
    max_value = df_temp.iloc[df_temp['expected_value'].idxmax(), 1]
    
    if return_type == 'scalar':
        return (max_value_thresh, max_value)
    elif return_type == 'graph':
        plt.figure(figsize=figsize)
        
        if show_positive_only is True:
            df_temp = df_temp.query('expected_value >= 0')
            sns.lineplot(data=df_temp, x='prob_threshold', y='expected_value')
        else:
            sns.lineplot(data=df_temp, x='prob_threshold', y='expected_value')
        
        plt.axvline(max_value_thresh, color='grey', ls='--', alpha=0.3, label=f"thresold: {max_value_thresh}")
        plt.axhline(max_value, color='red', alpha=0.3, label=f"max_expected_value: {max_value}")
        title_text = f'Expected value by each probability threshold ({df_temp.shape[0]}) | show_positive_values_only = {show_positive_only}'
        plt.title(title_text)
        plt.legend(bbox_to_anchor=(1,1))
        plt.show()
    else:
        raise Exception("return_type must be either 'scalar', or 'graph'!")

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
    
    df_feature = pd.DataFrame({'feature_importance': tree_predictor.feature_importances_ * 100},
                              index=tree_predictor.feature_names_in_)
    
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
    plt.xlabel(f'Total amount {criterion} decreased by splits in % (averaged over all trees if RF)', fontsize='small')
    plt.show()