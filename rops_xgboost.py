import feather
import sklearn
import xgboost
import numpy as np
import pandas as pd

import json

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set(style="darkgrid")

df = feather.read_dataframe('/home/jovyan/data/rops/28f8c94651a8e36544519396faca9a99/manestream/v2/baseline/v1/df.feather')

schema_path = '/home/jovyan/data/rops/28f8c94651a8e36544519396faca9a99/manestream/v2/baseline/v1/schema.json'
with open(schema_path, 'r') as f:
    schema = json.load(f)

inverse_mapper = schema['category']['inverse_mapper'].copy()

for k, v in inverse_mapper.items():
    inverse_mapper[k] = {int(kk): vv for kk, vv in v.items()} # have to make sure keys are int for remapping

cats = schema['category']['mapper'].keys()
for cat in cats:
    df[cat] = df[cat].map(inverse_mapper[cat]) 

#df.head()

print(df.shape)
df = df[df['excluded_reason'] == 'missing']
df['dos_age'] = df['dos_from_age'] - df['dos_thru_age']
df['dos_age_bin'] = pd.cut(df['dos_age'], [0, 5, 10, 15, 20, 25, 30], include_lowest=True)
print(df.groupby('dos_age_bin')['residual_ae'].describe().reset_index().T)
#facet_by('dos_age_bin', cw=4, top=0.9)

save_df = df.reset_index()
save_df['dos_age_bin'] = save_df['dos_age_bin'].astype(str)
save_df.to_feather('/home/jovyan/use_df.feather')

print(df.filter(like='fatal').columns)

qoi = {
    'informative': [
        'report_date',
        'delta_date',
        'claim_no'
    ],
    'category': [
        'dataset',
        'active_queue',
        'coverage_level',
        'last_transaction',
        'insurance_product',
        'insurance_state',
        'r_code',
        'prt_status',
        'excluded_reason',
        'fatal_denial_untimely', 'fatal_denial_cob', 'fatal_denial_non_covered',
       'fatal_denial_auth', 'fatal_denial_appeal_denied',
       'fatal_denial_setup'
    ],
    'continuous': [
        'gross_charges',
        'net_revenue',
        'receipts',
        'adjustments',
        'bad_debt_reserve',
        'balance',
        'last_remit_pat_resp',
        'research_cash',
        'last_transaction_date_age',
        'dos_thru_age',
        'date_of_transfer_age',
        'gen_date_age',
        'billing_date_age',
        'time_to_file_age',
        'dos_age'
    ],
    'target': [
        'receipts_delta'
    ]
}

df1 = df[df['receipts_delta'] != 0.]

df = df.dropna(subset=qoi['category'] + qoi['continuous'] + qoi['target'] + ['delta_date'])
print(df.shape)
from sklearn.model_selection import train_test_split
print('After dropna:', df.shape)
df['receipts_delta'] = df['receipts_delta'].fillna(0)

#df = df[~df['receipts_delta'].isnull()]

print(df.shape)
print(df[df['receipts_delta'].isnull()])
print(df[df['receipts_delta'] < 0.].shape)
df[df['receipts_delta'] > 0.].shape

df = df[df['receipts_delta'] <= df['receipts_delta'].mean() + 5*df['receipts_delta'].std()]
df = df[df['receipts_delta'] >= df['receipts_delta'].mean() - 5*df['receipts_delta'].std()]
print(df.shape)


cont_df = df[qoi['continuous']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(cont_df.values)
cont_df = pd.DataFrame(scaler.transform(cont_df), columns=qoi['continuous'])

%matplotlib inline

df_hist = df.filter(like='receipts_delta').hist()

print(df_hist)

from sklearn.metrics import mean_squared_error, mean_absolute_error

cat_df = pd.get_dummies(df[qoi['category']].astype(str))
date_df = pd.get_dummies(df['delta_date'].astype(int))
target_df = df.filter(like='receipts_delta')

cat_df.reset_index(drop=True, inplace=True)
date_df.reset_index(drop=True, inplace=True)
target_df.reset_index(drop=True, inplace=True)
cont_df.reset_index(drop=True, inplace=True)

print(cont_df.shape, cat_df.shape, date_df.shape, target_df.shape)


full_df = pd.concat([cont_df, cat_df, date_df], axis=1)

#full_df.shape

from sklearn.model_selection import GroupKFold

X = full_df.values
y = target_df.values

groups = df['claim_no'].values
gkf = GroupKFold(n_splits=5)
gkf.get_n_splits(X, y, groups)


X = np.concatenate([cont_df.values, cat_df.values, date_df.values], axis=1)
y = df[qoi['target']].values
print('X shape:', X.shape)
print('y shape:', y.shape)

import xgboost as xgb

def get_metrics(X, y):
    yhat = clf.predict(X)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae = mean_absolute_error(y, yhat)

    return rmse, mae

error_dict = {}
errors = {}
errors['train'] = {}
errors['val'] = {}

clf = xgb.XGBRegressor(n_estimators=25, tree_method='hist', 
                       subsample=0.9, max_depth=15, learning_rate=0.1, eval_metric=['mae', 'rmse'])

for idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print('Length of train idx:', len(train_idx))
    print('Length of val idx:', len(val_idx))
    train_X, val_X = X[train_idx], X[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]
    
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (val_X, val_y)], early_stopping_rounds=15)
    train_rmse, train_mae = get_metrics(train_X, train_y)
    val_rmse, val_mae = get_metrics(val_X, val_y)
    errors['train']['rmse'] = train_rmse
    errors['train']['mae'] = train_mae
    errors['val']['rmse'] = val_rmse
    errors['val']['mae'] = val_mae
    error_dict[f'run_{idx+1}'] = errors
    break
print(f'Train rmse:', train_rmse, 'train mae:', train_mae)
print(f'Val rmse:', val_rmse, 'val mae:', val_mae)

print(f'All errors:')
print(error_dict)


yhat = clf.predict(X[val_idx])
val_df = df.ix[val_idx]
val_df['yhat'] = yhat
val_df['y'] = y[val_idx]
val_df['residual'] = val_df['y'] - val_df['yhat']
val_mean = val_df['residual'].mean()
val_df['residual_ae'] = np.abs(val_df['residual'])


print(len(yhat), val_X.shape)

print(val_df.shape)

def facet_by(fby, qoi='residual', val_df=val_df, xlo=-2000, xhi=2000, cw=None, norm_hist=False, sharey=False, top=0.8):
    mdf = val_df[[qoi, fby]].melt([qoi], var_name='cols',  value_name='vals')
    g = sns.FacetGrid(mdf, col='vals', palette="Set1", col_wrap=cw, sharey=sharey).set(xlim=(xlo, xhi))
    g = (g.map(sns.distplot, qoi, norm_hist=norm_hist, kde=False, bins=100))
    g.set_titles(col_template='{col_name}')

    yfmt = '{x:,.0f}'
    ytick = mtick.StrMethodFormatter(yfmt)
    
    for ax in g.axes.flatten():
        ax.axvline(val_mean, ls='--', color='r')
        ax.set_xlabel('')
        if norm_hist == False:
            ax.yaxis.set_major_formatter(ytick) 
    g.fig.suptitle(f'{qoi.capitalize()} by {" ".join([x.capitalize() for x in fby.split("_")])}', size=16)
    g.fig.subplots_adjust(top=top)
    return g
                   
#myPlot = myPlot.map(plt.scatter, "x", "y").set(xlim=(-20,120) , ylim=(-15,15))

print(val_df.groupby('financial_class')['residual_ae'].describe().reset_index().T)
facet_by('financial_class', xhi=4000)

val_df['residual_ae'].tail()
val_df[['y', 'yhat', 'residual', 'residual_ae']].describe()


fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sns.distplot(val_df['residual'], bins=100, kde=False).set(xlim=(-5000, 5000))

fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.xaxis.set_major_formatter(tick)

yfmt = '{x:,.0f}'
ytick = mtick.StrMethodFormatter(yfmt)
ax.yaxis.set_major_formatter(ytick)
ax.set_title('Residuals for Receipts Prediction (Actual - Predicted)')
ax.set_xlabel('Error in Dollar Amount')

fig, ax = plt.subplots(1, 1, figsize=(10, 6))



sns.distplot(val_df['y'], norm_hist=True, kde=False, bins=50).set(xlim=(-3000, 6000))
sns.distplot(val_df['yhat'], norm_hist=True, kde=False, bins=50).set(xlim=(-3000, 6000))

fig.legend(labels=['Actual Receipts','Predicted Receipts'])
plt.xlabel('')
plt.ylabel('Count')
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.xaxis.set_major_formatter(tick)

yfmt = '{x:,.0f}'
ytick = mtick.StrMethodFormatter(yfmt)
ax.yaxis.set_major_formatter(ytick)

#plt.xticks(rotation=0)

#ax.set_title('Actual vs Predicted Comparison')
#ax.set_xlabel('Receipts Amount')

print(val_df.groupby('financial_class')['residual_ae'].describe().reset_index().T)
#facet_by('financial_class', xhi=4000)

print(val_df.groupby('coverage_level')['residual_ae'].describe().reset_index().T)
f#acet_by('coverage_level', xhi=4000)

val_df['receipts_zero_flag'] = np.where(val_df['receipts'] == 0, 'Yes', 'No')
print(val_df.groupby('receipts_zero_flag')['residual_ae'].describe().reset_index().T)
#facet_by('receipts_zero_flag', xhi=4000)

val_df['receipts_delta_zero_flag'] = np.where(val_df['receipts_delta'] == 0, 'Yes', 'No')
print(val_df.groupby('receipts_delta_zero_flag')['residual_ae'].describe().reset_index().T)
#facet_by('receipts_delta_zero_flag', xhi=4000)


val_df['receipts_delta_change_flag'] = np.where((val_df['receipts'] == 0.) & (val_df['receipts_delta'] == 0.), 'Yes', 'No')
print(val_df.groupby('receipts_delta_change_flag')['residual_ae'].describe().reset_index().T)
#facet_by('receipts_delta_change_flag', xhi=4000)

print(val_df.groupby('insurance_product')['residual_ae'].describe().reset_index().T)
#facet_by('insurance_product', cw=4, top=0.9)

dd = val_df.groupby('last_transaction')['residual_ae'].describe().reset_index().T
ddd = val_df.groupby('last_transaction')['residual'].describe().reset_index().T
out = []
outt = []
cols = []
for col in dd:
    if dd.loc['count', col] < 1000:
        continue
    else:
        out.append(dd[col])
        outt.append(ddd[col])
        cols.append(col)
print_df = pd.concat(out, axis=1)
plot_df = val_df[val_df['last_transaction'].isin(print_df.ix[0])]

print(print_df)


#facet_by('last_transaction', val_df=plot_df, cw=4, top=0.9)

print(val_df.groupby('coll_map')['residual_ae'].describe().reset_index().T)

#facet_by('coll_map')

print(val_df.groupby('npsp_type')['residual_ae'].describe().reset_index().T)

#facet_by('npsp_type')

sssf = val_df[['residual', 'gross_charges', 'net_revenue', 'receipts', 'adjustments']]

#sns.pairplot(sssf)

#fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sssf = val_df[['residual', 'receipts_delta', 'receipts', 'receipts_next']] # 'net_revenue_delta', 'adjustments_delta']]
yfmt = '{x:,.0f}'
ytick = mtick.StrMethodFormatter(yfmt)

g = sns.pairplot(sssf)

yfmt = '{x:,.0f}'
ytick = mtick.StrMethodFormatter(yfmt)

for ax in g.axes.flatten():
    ax.xaxis.set_major_formatter(ytick) 
    ax.yaxis.set_major_formatter(ytick) 


sssf = val_df[['residual', 'billing_date_age', 'dos_thru_age', 
               'last_transaction_date_age', 'date_of_transfer_age',
              'eresponse_status_code_age']]
yfmt = '{x:,.0f}'
ytick = mtick.StrMethodFormatter(yfmt)

g = sns.pairplot(sssf)

yfmt = '{x:,.0f}'
ytick = mtick.StrMethodFormatter(yfmt)

for ax in g.axes.flatten():
    ax.xaxis.set_major_formatter(ytick) 
    ax.yaxis.set_major_formatter(ytick) 

val_df['dos_age'] = val_df['dos_from_age'] - val_df['dos_thru_age']
val_df['dos_age_bin'] = pd.cut(val_df['dos_age'], [0, 5, 10, 15, 20, 25, 30], include_lowest=True)
print(val_df.groupby('dos_age_bin')['residual_ae'].describe().reset_index().T)
facet_by('dos_age_bin', cw=4, top=0.9)