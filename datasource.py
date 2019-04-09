from functools import partial
from datetime import timedelta
import time
from oai.db import nz

from typing import Optional

from oai.datasource import DataSource

import pandas as pd
import numpy as np

from oai import logger

log = logger.global_logger(__name__)


def transfer(query=None) -> pd.DataFrame:
    if not query:
        df_list = []
        report_date_ranges = [str(x).split(' ')[0] for x in pd.date_range('2018-01-01', '2019-01-31', freq='M')]

        for idx, rdr in enumerate(report_date_ranges):

            query = f"""
            SELECT * 
              FROM datalab_iff_1..claims_delta
             WHERE append_date = '{rdr}'
               AND financial_class IN ('COMM', 'CONTRT')
            """

            if idx == 0:
                print('')
                time.sleep(0.1)
            log.info(f'\n\nCurrently being pulled...\n{query}')
            
            tmp_df = nz.to_df(query)
            df_list.append(tmp_df)

            stars = '*'*100

            log.info(f'\n\nFinished batch pull for append_date = {rdr}\n\n{stars}\n{stars}\n')
            
        df = pd.concat(df_list)

        log.info(f'Finished data concatenation\n\n{stars}\n{stars}\n')
        
    else:
        df = nz.to_df(query)

    return df

def engineer(df: pd.DataFrame, age_columns) -> pd.DataFrame:
    pre_names = df.columns.tolist()
    df = df.pipe(create_age_features, date_froms=age_columns)
    new_columns = list(set(df.columns.tolist()) - set(pre_names))
    retval = {}
    retval['df'] = df
    retval['column_types'] = {'int32': df[new_columns].columns.tolist()}
    return retval

def create_age_features(df, date_froms, to_date='delta_date'):
    '''Output df with columns giving difference in days between list of dates given in date_froms'''
    if not to_date:
        to_date = datetime.datetime.today().strftime('%Y%m%d')
    if to_date == 'delta_date':
        to_date = df['delta_date']
    for date_from in date_froms:
        column_name = f'{date_from}_age'
        df[column_name] = (pd.to_datetime(to_date) - pd.to_datetime(df[date_from], errors ='coerce')).dt.days.fillna(0).astype(np.int32)
    return df



xy_diff = lambda x, y: len(set(x) - set(y))

def get_previous_diff(df, current_date, column_name, debug=False):
    cd = pd.to_datetime(current_date)
    current_date = current_date
    previous_date = f"{pd.to_datetime(cd.replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')}"
    prev_d = pd.to_datetime(previous_date)
    previous = df.loc[df['report_date'] == prev_d, column_name]
    current = df.loc[df['report_date'] == cd, column_name]
    dequeued_ids = set(previous) - set(current)
    queued_ids = set(current) - set(previous)
    queued_count = xy_diff(previous, current)
    dequeued_count = xy_diff(current, previous)
    return list(dequeued_ids), list(queued_ids), previous_date, current_date

def report_date_deltas(df):
    doi = np.sort(list(pd.to_datetime(df['report_date']).astype(str).unique()))
    diff_dict = {k: partial(get_previous_diff, current_date=k, column_name='claim_no', df=df) for k in doi}
    dequeued_list = []
    queued_list = []

    for idx, d in enumerate(doi):
        if idx == 0:
            date_prev = d
            continue
        dequeued_idx, queued_idx, previous_date, current_date = diff_dict[d]()
        
        # Add worked claim_no to previous report_date
        dequeued_df = pd.DataFrame({'claim_no': dequeued_idx})
        dequeued_df['report_date'] = pd.to_datetime(date_prev)
        dequeued_df['dequeued'] = 'Y'

        queued_df = pd.DataFrame({'claim_no': queued_idx})
        queued_df['report_date'] = pd.to_datetime(d)
        queued_df['queued'] = 'Y'
    
        dequeued_list.append(dequeued_df)
        queued_list.append(queued_df)
        log.info(queued_df.shape)
        log.info(dequeued_df.shape)
        log.info(f'Report month {d} had {len(dequeued_idx)} claims dequeued and report month {date_prev} had {len(queued_idx)} claims queued')
    
        date_prev = d
    
    if len(dequeued_list) == 0:
        df['queued'] = 'N'
        df['dequeued'] = 'N'
        df['queued'] = df['queued'].astype('category')
        df['queued'] = df['queued'].astype('category')
    else:
        dequeued_df = pd.concat(dequeued_list)
        queued_df = pd.concat(queued_list)

        df = (df.merge(dequeued_df, how='left', on=['claim_no', 'report_date'])
                .merge(queued_df, how='left', on=['claim_no', 'report_date']))

        df[['queued', 'dequeued']] = df[['queued', 'dequeued']].fillna('N').astype('category')
    return df