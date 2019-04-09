from oai.datapackager import DataPackager

from oai.tabular.datasets.boilerplate import DualHeadDataset

import pandas as pd

def by_date_strategy(self, df: pd.DataFrame) -> dict:
    sorted_dates = df['report_date'].sort_values()

    test_date = sorted_dates[0]
    val_date = sorted_dates[1]
    learn_dates = sorted_dates[1:]

    df_dict = {}

    df_dict['test'] = df[df['report_date'] == test_date]
    df_dict['val'] = df[df['report_date'] == val_date]
    df_dict['learn'] = df[df['report_date'].isin(learn_dates)]
    return df_dict

class DequeuedDataPackager(DataPackager):
    """docstring for DequeuedDataPackager"""
        
    def dataset(self, df: pd.DataFrame):
        return DualHeadDataset(df=df, cat_cols=self.cat_cols, cont_cols=self.cont_cols, target_col=self.target_col)

    def strategy(self, df: pd.DataFrame) -> dict:
        sorted_dates = df['report_date'].sort_values()

        test_date = sorted_dates[0]
        val_date = sorted_dates[1]
        learn_dates = sorted_dates[1:]

        df_dict = {}

        df_dict['test'] = df[df['report_date'] == test_date]
        df_dict['val'] = df[df['report_date'] == val_date]
        df_dict['learn'] = df[df['report_date'].isin(learn_dates)]
        return df_dict


def ShallowArchitecture():
    pass

def ClassificationLearner():
    pass

#def DeququedExperiments(Experiments):
#    pass