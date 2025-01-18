"""

Created on Mon Jul 22 18:29:22 2024

@author: Hao
"""
##### IMPORTS #####
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

import my_funcs_prep as my  # personal functions.

##### LOADING DATA #####
df = pd.read_csv('./data-new/transactions_obf.csv')
labels = pd.read_csv('./data-new/labels_obf.csv')

# Create target fraud flag in main df.
fraud_eventId = labels.eventId
df['isFraud'] = np.where(df.eventId.isin(fraud_eventId), 1, 0)
# Imbalanced classes: 875 fraud transactions out of 117746.
df['isFraud'].value_counts(normalize=True)

# Check duplicates.
df.duplicated().sum()

# mcc, merchantCountry, posEntryMode are actually categorical variables.
df.info(verbose=True)
df = df.astype({
    'mcc' : 'str',
    'merchantCountry' : 'str',
    'posEntryMode' : 'str',
})

# Change transactionTime to datetime dtype.
df['transactionTime'] = pd.to_datetime(df.transactionTime, utc=True)

##### UNIVARIATE ANALYSIS #####
# Check number of unique values for each variable.
df.nunique().sort_values(ascending=True)
num_vars = ['availableCash', 'transactionAmount']
cat_vars = [
    'posEntryMode', 'accountNumber', 
    'merchantId', 'mcc', 'merchantCountry', 'merchantZip'
]

# Categorical variables.
# Show prop and count for each category in descending order for each categorical variable.
for cat in cat_vars:
    cat_prop  = df[cat].value_counts(normalize=True, sort=False, dropna=False).round(3)
    cat_count = df[cat].value_counts(normalize=False, sort=False, dropna=False)
    df_frequency = pd.concat([cat_prop, cat_count], axis=1).sort_values(by='proportion', ascending=False)
    print(df_frequency, '\n')

# Group infrequent categories as 'other' in posEntryMode and merchantCountry.
df['posEntryMode_red'] = my.group_infreq_labels(
    df['posEntryMode'],
    # the relative frequecny threshold below which labels are considered infrequent.
    threshold=0.08,
    label='other'
)
df['merchantCountry_red'] = my.group_infreq_labels(
    df['merchantCountry'],
    threshold=0.1,
    label='other'
)

'''
OBSERVATIONS:

- posEntryMode and merchantCountry exhibit a few predominant categories that 
constitute approximately 90% of all observations. I consider grouping 
the remaining infrequent categories together.
    - posEntryMode: 5 (59%), 81 (30%), 1 (9%)...
    - merchantCountry: 826 (80%), 442 (13%), ...

- For the other three categorical variables, applying frequency encoding might
be more appropriate. This would account for every category since there 
are numerous categories (100+) with similar, small relative proportions.
'''

# Numeric variables.
df[num_vars].describe()\
            .round(2)\
            .transpose()

# Exploring negative transactions.
df[df.transactionAmount < 0].shape[0]
df[df.transactionAmount < 0]['transactionAmount'].describe()

'''
OBSERVATIONS:

- The distribution of transactionAmount is heavily skewed to the right
as mean (53.67) > median (20.25). This indicates the presence of many large
transaction amounts. Interestingly, there are 183 negative transactions.
These negative amounts do not appear to be recording errors since they all are
less than 0.15 currency units and occur frequently.Discounts? Fee adjusments?
'''

##### HANDLING MISSING VALUES #####
df.isna()\
  .mean()\
  .mul(100)\
  .sort_values(ascending=False)

# missing values only in merchantZip: 19.4%. Replace by 'Unknown'.
df['merchantZip'] = df['merchantZip'].replace(to_replace=np.nan, value='Unknown')

##### FEATURE ENGINEERING #####
# Create basic time-based features to capture temporal patterns.
df['year'] = df['transactionTime'].dt.year
df['month'] = df['transactionTime'].dt.month
df['day'] = df['transactionTime'].dt.day

##### EDA #####
# Numeric variables vs y (interactive plots).
for num in num_vars:
    fig = my.plot_box_byCat(df, num, 'isFraud', log_y=True)
    fig.show()

# Categorical variables (low cardinality) vs y
cat_vars_few_cats = [
    'posEntryMode_red', 'merchantCountry_red',
    'year', 'month', 'day'
]

for cat in cat_vars_few_cats:
    fig = my.plot_freq_barh_byCat(df['isFraud'], df[cat])
    fig.show()

'''
OBSERVATIONS:

- On average, fraudulent transactions tend to involve larger amounts than
 legitimate transactions, partly due to the presence of many positive outliers,
 with averages of 115 and 53, respectively. Additionally, fraudulent
 transactions typically are more variable and have lower available cash, 
 with a average of 4,575 compared to 6640 for legitimate transactions.

- In terms of categorical variables, fraudulent transactions are more
 frequently associated with posEntryMode values of 1 and 81, and occur more
 often in merchant countries other than 826 and 442. They are also more likely
 to take place in the months of June, September, and October, during the days
 17-23.
'''

#####  Analyze temporal patterns in fraud transactions #####
# Compare number of transactions per accountNumber by isFraud.
df_fraud = df[df['isFraud'] == 1]
accounts_with_fraud = df_fraud['accountNumber'].unique()
df_accounts_with_fraud = df.loc[df.accountNumber.isin(accounts_with_fraud)]
df_accounts_no_fraud = df.loc[~df.accountNumber.isin(accounts_with_fraud)]

df_accounts_with_fraud_count = df_accounts_with_fraud.groupby('accountNumber')\
                                                     ['transactionAmount']\
                                                     .count()\
                                                     .reset_index(name='count')

df_accounts_no_fraud_count = df_accounts_no_fraud.groupby('accountNumber')\
                                                 ['transactionAmount']\
                                                 .count()\
                                                 .reset_index(name='count')

fig = my.plot_num_hist_box(df_accounts_with_fraud_count, 'count', title='Fraud account count')
fig.show()

fig = my.plot_num_hist_box(df_accounts_no_fraud_count, 'count', title='No fraud account count')
fig.show()

'''
OBSERVATIONS:

- Let's compare the median values since both count distributions are
right-skewed. On median, accounts with fraudulent transactions make more
transactions than accounts without, with medians of 101 and 63, respectively.
'''

# Search for fraud transaction patterns in 10 random accounts with fraud transations.
for i, row in df_accounts_with_fraud_count.sample(10).iterrows():
    df_fraud_account = df[df.accountNumber == row['accountNumber']]
    df_fraud_transactions = df_fraud_account.loc[df_fraud_account.isFraud == 1]
    fig = px.line(
        df_fraud_account,
        x='transactionTime',
        y='transactionAmount',
        markers=True,
        template='plotly_white',
        hover_data={'merchantCountry':True, 'merchantZip':True, 'merchantId':True},
        title=f"Fraud account: {row['accountNumber']} | Number fraud tnx: {len(df_fraud_transactions)}/{row['count']}"
    )

    fig.add_trace(go.Scatter(
        mode='markers',
        name='Fraud',
        text=df_fraud_transactions['merchantCountry'] + ' ' + df_fraud_transactions['merchantZip'] + ' ' + df_fraud_transactions['merchantId'],
        x=df_fraud_transactions['transactionTime'],
        y=df_fraud_transactions['transactionAmount'],
    ))

    fig.show()

# Observe the average count of fraudulent transactions per account 
# with at least one transaction flagged as fraud. 
df_fraud_tnx_per_account_count = df_fraud.groupby('accountNumber')\
                                         ['isFraud']\
                                         .count()\
                                         .reset_index(name='count')

fig = my.plot_num_hist_box(df_fraud_tnx_per_account_count, 'count', title='Fraud tnx per account')
fig.show()

'''
OBSERVATIONS:

- After observing many accounts with fraudulent transactions over time (time series of transaction amounts), I noticed a plethora of diversity in their patterns:
    - Several fraudulent transactions occur within a span of or hours, minutos 
    or even seconds on the same day, while some happen on different days.
    - Extreme case for accountNumber '4cfd251e' where the only transaction made was a fraudulent one!
    - There are accounts with just a single fraudulent transaction.
    - Fraudulent transactions can have amounts similar to previous legitimate
     ones, while others have unusually high amounts.
    - Several fraudulent transactions on same merchant and place.

- On median, accounts with fraudulent trasactions have 5 transactions flagged as fraud. Based on all this information, let's create some features that capture more complex transaction patterns for each accountNumber.
'''

##### ADVANCED FEATURES #####
# Number of transactions per account.
df_accounts_count = df.groupby('accountNumber')\
                      ['transactionAmount']\
                      .count()\
                      .reset_index(name='tnx_count_account')
df = df.merge(right=df_accounts_count, how='inner', on='accountNumber')

# Transaction count per day and account. 
# temp_ cols are intermediate cols to be eliminated later on.
df['temp_date'] = df['transactionTime'].dt.date
df_daily_tnx_count = df.groupby(['accountNumber', 'temp_date'])\
                       ['temp_date']\
                       .count()\
                       .reset_index(level=0, name='tnx_count_daily')\
                       .reset_index() # get date index back as column!
df = df.merge(right=df_daily_tnx_count, how='inner', on=['accountNumber', 'temp_date'])

# Seconds past since last transaction per account.
df = df.sort_values('transactionTime')
df['seconds_since_last_tnx'] = df.groupby('accountNumber')\
                                 ['transactionTime']\
                                 .diff(periods=1)\
                                 .dt.total_seconds()
# Fill first NaN obervation for each account by the median per account.
median_s_since_last_tnx = df.groupby('accountNumber')\
                            ['seconds_since_last_tnx']\
                            .transform('median')
df['seconds_since_last_tnx'] = df['seconds_since_last_tnx'].fillna(median_s_since_last_tnx)
# There are 11 accounts with only one transaction: NaN. Let's fill them with 0.                                 
(df_accounts_count['tnx_count_account'] <= 1).sum()
df['seconds_since_last_tnx'] = df['seconds_since_last_tnx'].fillna(0)

# Get 7-tnx rolling min, max, avg.
df['tnx_rolling7_max'] = df.groupby('accountNumber')\
                           ['transactionAmount']\
                           .rolling(7, min_periods=1)\
                           .max()\
                           .reset_index(level=0, drop=True) # removes extra indices

df['tnx_rolling7_min'] = df.groupby('accountNumber')\
                           ['transactionAmount']\
                           .rolling(7, min_periods=1)\
                           .min()\
                           .reset_index(level=0, drop=True) 

df['tnx_rolling7_avg'] = df.groupby('accountNumber')\
                           ['transactionAmount']\
                           .rolling(7, min_periods=1)\
                           .mean()\
                           .reset_index(level=0, drop=True)

df['tnx_to_rolling_avg_ratio'] = df['transactionAmount'] / df['tnx_rolling7_avg']
# zero division cases replace with 0.
df['tnx_to_rolling_avg_ratio'] = df['tnx_to_rolling_avg_ratio'].fillna(0)

# Dummies for transaction with same transactionAmount, location 
# or/and merchant than previous one per account.
df['same_prev_tnx'] = df['transactionAmount'] == df.groupby('accountNumber')\
                                                   ['transactionAmount']\
                                                   .shift(1)

df['same_prev_merch_id'] = df['merchantId'] == df.groupby('accountNumber')\
                                                 ['merchantId']\
                                                 .shift(1)

# same location means same merchantCountry and merchantZip.
df['temp_location'] = df['merchantCountry'] + df['merchantZip']
df['same_prev_merch_loc'] = df['temp_location'] == df.groupby('accountNumber')\
                                                     ['temp_location']\
                                                     .shift(1)

##### ENCODING CATEGORICAL VARIABLES #####
# convert booleans to int8 (1 and 0).
bool_vars = ['same_prev_tnx', 'same_prev_merch_id', 'same_prev_merch_loc']
df[bool_vars] = df[bool_vars].astype('int8')

# One Hot Encoding ofr some cat variables.
ohe_cat_vars = [
    'merchantCountry_red', 'posEntryMode_red',
    'month', 'day', 'year',
]
df = pd.get_dummies(df, columns=ohe_cat_vars, dtype='int8', drop_first=False)

# Create frequency encoded versions for some cat variables.
freq_cat_vars = ['mcc', 'merchantZip', 'merchantId']
for cat in freq_cat_vars:
    dict_freqs = df[cat].value_counts(normalize=True, sort=True).to_dict()
    df[f"{cat}_freq"] = df[cat].map(dict_freqs)

# Eliminate variables.
to_delete_vars = [
    'posEntryMode', 'merchantCountry',
    'temp_date', 'temp_location', 'accountNumber',
    'mcc', 'merchantZip', 'merchantId'
]
# We end up with time-ordered df with 118,621 rows and 69 cols.
df = df.drop(columns=to_delete_vars)

##### TRAIN-TEST DATA SPLIT #####
# 2017-01 to 2017-12 as training set and 2018-01 as test set.
df = df.set_index('transactionTime').sort_index()

df_train = df.loc[:'2017-12'].reset_index(drop=True)  # Get rid of datetime index.
df_test  = df.loc['2018-01':].reset_index(drop=True)

X_train = df_train.drop(columns=['isFraud', 'eventId'])
y_train = df_train['isFraud']

# Save eventId.
eventId_test = df_test.eventId
X_test  = df_test.drop(columns=['isFraud', 'eventId'])
y_test  = df_test['isFraud']

'''
OBSERVATIONS:

- I implied that the client wants to predict transactions in a monthly basis as
 they review 400 transactions monthly. That explains why I chose the last month as test set. 
'''
##### MODEL SELECTION #####
# Use Timeseries split to mantain temporal order in training data.
ts_cv = TimeSeriesSplit(n_splits=5)

def get_best_randsearch_model(
    estimator,
    param_dict,
    X,
    y
):
    '''
    Returns a df with randomized search results: mean_fit_time, mean_train_score, 'mean_test_score, and params.
    '''
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dict,
        n_iter=30,
        scoring='f1',
        n_jobs=-1, 
        cv=ts_cv,
        refit=True,
        random_state=rand_state,
        return_train_score=True
    )

    random_search.fit(X, y)
    cols = ['mean_fit_time', 'mean_train_score', 'mean_test_score'] + [f"param_{k}" for k in param_dict.keys()]
    df_results = pd.DataFrame(random_search.cv_results_)
    df_results = df_results[cols].sort_values('mean_test_score', ascending=False)
    return df_results

##### BEST LOGISTIC REGRESSION  #####
# Set random state for reproducibility.
rand_state = 0
# Instantiate candidate models with default hyperparameters.
# We set balaced class weights for models to pay more attention to minority class.
logreg = LogisticRegression(
    random_state=rand_state,
    class_weight='balanced',
)

param_dict_logreg = {
    'penalty' : [None, 'l2'],
    'C'       : np.linspace(1, 100, 100),
    'solver'  : ['lbfgs', 'newton-cg', 'sag', 'saga']
}

logreg_results = get_best_randsearch_model(
    estimator=logreg,
    param_dict=param_dict_logreg,
    X=X_train,
    y=y_train
)

##### BEST RANDOM FOREST CLASSIFIER  #####
rfc = RandomForestClassifier(
    random_state=rand_state,
    class_weight='balanced'
)

param_dict_rfc = {
    'n_estimators'     : np.arange(50, 80, 10),
    'max_depth'        : np.arange(1, 8, 1),
}

rfc_results = get_best_randsearch_model(
    estimator=rfc,
    param_dict=param_dict_rfc,
    X=X_train,
    y=y_train
)

##### BEST LIGHTGBM CLASSIFIER  #####
lgbmc = LGBMClassifier(
    objective='binary',
    verbose=-1,
    random_state=rand_state,
    class_weight='balanced'
)

param_dict_lgbmc = {
    'n_estimators' : np.arange(50, 110, 10),
    'max_depth'    : np.arange(1, 5, 1),
    'learning_rate': np.linspace(0.01, 0.1, 20)
}

lgbmc_results = get_best_randsearch_model(
    estimator=lgbmc,
    param_dict=param_dict_lgbmc,
    X=X_train,
    y=y_train
)

'''
OBSERVATIONS:
- Chose f1 score as evaluation metric as I'm interested in detecting fraud transactions and care about both precision and recall in the minority class, (y=1).
- For each model I chose the combination of hyperparemeters that gave highest f1 score in mean_test_score provided there is no strong overfitting comparing it with mean_train_score.

- best LogisticRegression params: {'solver': 'newton-cg', 'penalty': 'l2', 'C': 77.0}
- best LogisticRegression fit-time: 17.37
- best LogisticRegression train_f1: 0.078
- best LogisticRegression   val_f1: 0.089

- best RandomForestClassifier params: {'n_estimators': 50, 'max_depth': 7}
- best RandomForestClassifier fit_time: 4.68
- best RandomForestClassifier train_f1: 0.12
- best RandomForestClassifier   val_f1: 0.11

- best LightGBMClassifier params: {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
- best LightGBMClassifier fit_time: 1.1
- best LightGBMClassifier train_f1: 0.115
- best LightGBMClassifier   val_f1: 0.108

- I'll choose the LightGBMClassifier as the final model for several reasons:
it trains faster, the difference in F1 score between the RandomForestClassifier
and LightGBMClassifier is marginal, and the train and validation scores for
LightGBM are more similar to each other, indicating lower variance and potentially better generalization.
'''

###### PERFORMANCE EVALUATION #####
# Best LightGBMClassifier model.
lgbmc_rdsearch = LGBMClassifier(
    objective='binary',
    verbose=-1,
    random_state=rand_state,
    class_weight='balanced',
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1
)

lgbmc_rdsearch.fit(X_train, y_train)
print('Training set (LightGBMClassifier):')
y_pred_train = lgbmc_rdsearch.predict(X_train)
print(classification_report(y_true=y_train, y_pred=y_pred_train, digits=3))

print('Test set (LightGBMClassifier):')
y_pred = lgbmc_rdsearch.predict(X_test)
print(classification_report(y_true=y_test, y_pred=y_pred, digits=3))
ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred)

# Let's see what are the top 10 most important features.
my.feature_importance_plot(lgbmc_rdsearch, n_rows=10) 
df_feature_importance = my.feature_importance_plot(lgbmc_rdsearch, return_df=True) 

'''
OBSERVATIONS:

- The F1 score resulting from the tuned LightGBM using all training data is slightly higher on the training set than on the test set, with scores of 0.099 and 0.053, respectively. The model has identified 34 fraud transactions out of 39 (recall of 0.87) in a total of 8531 transactions. However, the precision is extremely low at 0.03, leading to 1219 false positives.

- The top 10 most important features based on feature importance provided by LGBM are:
    1.tnx_count_account.
    2.tnx_count_daily. 
    3.merchantId_freq.
    4.posEntryMode_red_5.
    5.transactionAmount.
    6.availableCash.
    7.tnx_rolling7_max.
    8.tnx_rolling7_avg.
    9.mmc_freq.
    10.tnx_rolling7_min.
'''

# Top 400 transactions that are most probable to be fraudulent in last month (test_set).
pred_proba = pd.Series(lgbmc_rdsearch.predict_proba(X_test)[:, 1], name='pred_proba')
df_final_pred = pd.concat([eventId_test, y_test, pred_proba], axis=1)
df_top_400_tnx = df_final_pred.sort_values(by='pred_proba', ascending=False)[:400]
df_top_400_tnx.loc[df_top_400_tnx.isFraud == 1].shape
df_top_400_tnx.loc[df_top_400_tnx.isFraud == 1].reset_index(drop=True)

'''
OBSERVATIONS:

- In the top 400 transactions most likely to be fraudulent during January 2018, 24 out of 39 (62%) fraudulent transactions were correctly identified!
'''