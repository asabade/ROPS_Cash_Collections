def insurance_product_value_function(df):
    df['insurance_product_value'] = df['balance']
    return df
def plan_group_value_function(df):
    df['plan_group_value'] = df['balance'] / 100
    return df

def center_code_value_function(df):
    df['center_code_value'] = df['balance']
    return df

def group_it(df, qoi, coi, agg_fct=np.mean, fill_value=0):
    gby_df = df.groupby(coi).agg(agg_fct)[qoi].fillna(fill_value).reset_index()
    return gby_df
    
def add_value(df, qoi, value_fct):
    out_df = df.pipe(value_fct)
    return out_df

def flag_anchor_point(df, value_column, coi):
    df.loc[:, f'{coi}_anchor_flag'] = False
    df.loc[df[value_column].idxmax, f'{coi}_anchor_flag'] = True
    return df

def calc_similarity(df, sim_values, anchor_column, metric):
    print(anchor_column)
    print(df)
    print('1'*80)
    anchor_point = df.loc[df[f'{anchor_column}_anchor_flag'], anchor_column].values[0]
    
    out = {}
    all_values = df[anchor_column].unique()
    compare_values = [x for x in all_values if x != anchor_point]
    df[f'{anchor_column}_diff_from_anchor'] = 0.0
    for cv in compare_values:
        df.loc[df[anchor_column] == cv, f'{anchor_column}_diff_from_anchor'] =\
            metric(df.loc[df[anchor_column] == cv, sim_values], 
                   df.loc[df[anchor_column] == anchor_point, sim_values])
    df = df.sort_values(f'{anchor_column}_diff_from_anchor').reset_index(drop=True)
    df[f'{anchor_column}_order'] = df.index + 1
    
    return df

def pipe_it(df, coi, qoi, value_function):
    print('#'*80)
    print(coi, qoi, value_function)
    out_df = (df.pipe(group_it, qoi=qoi, coi=coi)
                .pipe(add_value, qoi=qoi, value_fct=value_function)
                .pipe(flag_anchor_point, value_column=f'{coi}_value', coi=coi)
                .pipe(calc_similarity, sim_values=qoi, anchor_column=coi, metric=euclidean)
             )
    return out_df

def master_order(df, value_fcts):
    import pdb; pdb.set_trace()
    print('*'*80)
    print(value_fcts)
    output = {k: pipe_it(df=df, coi=k, qoi=qoi, value_function=v) for (k, v) in value_fcts.items()}
    
    print(cois)
    for i, coi in enumerate(cois):
        val = output[coi]
        val_cols = list(val.columns.difference(df.columns)) + [coi] 

        if i == 0:
            master_df = pd.merge(df, val[val_cols], on=coi, how='left')
        else:
            master_df = pd.merge(master_df, val[val_cols], on=coi, how='left')
    order_of_importance = ['insurance_product_order', 'plan_group_order', 'center_code_order']

    final_df = master_df.sort_values(order_of_importance).reset_index(drop=True)
    final_df['master_order'] = final_df.index + 1
    return final_df

from sklearn.preprocessing import StandardScaler

qoi = ['balance', 'net_revenue', 'research_cash']
cois = ['insurance_product', 'plan_group', 'center_code']
info_cols = ['claim_no', 
             'pod_name', 'append_date']
use_df = df[qoi + cois + info_cols].copy()
use_df[qoi] = use_df[qoi].apply(pd.to_numeric, errors='coerce').fillna(0)
use_df = use_df[use_df['append_date'] == '2018-02-28']
use_df = use_df[use_df['pod_name'] == 'INVICTA_POD_1']
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
use_df[qoi] = StandardScaler().fit_transform(use_df[qoi].fillna(0))

value_fcts = {x: globals()[f'{x}_value_function'] for x in cois}
print(value_fcts)
fdf = use_df.groupby(['pod_name', 'append_date']).apply(master_order, value_fcts=value_fcts)