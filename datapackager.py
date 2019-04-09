from abc import ABCMeta, abstractmethod
from typing import Optional
from importlib import import_module
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

def emb_half_or_50(cat_dims):
    emb_dims = [(x+1, min(50, (x + 1) // 2)) for x in cat_dims]
    return emb_dims

def emb_half_or_10(cat_dims):
    emb_dims = [(x+1, min(10, (x + 1) // 2)) for x in cat_dims]
    return emb_dims

class DataPackager(metaclass=ABCMeta):
    """docstring for DataPackager"""
    def __init__(self, 
                 df: pd.DataFrame, 
                 category: list, 
                 continuous: list,
                 dataset,
                 target: Optional[list], 
                 informative: Optional[list],
                 cat_dims: Optional[list],
                 emb_method: Optional[list],
                 strategy=None):
        super(DataPackager, self).__init__()
        self.df = df
        self.category = category
        self.continuous = continuous
        self.target = target
        self.informative = informative
        if self.target:
            self.output_size = len(self.target)
        self.no_of_cont = len(self.continuous)

        #self.cat_dims = cat_dims
        self.strategy = strategy
        self.dataset = dataset
        if self.strategy is not None:
            self.df_dict = self.strategy(self.df)
        else:
            self.df_dict = {}
            self.df_dict['inference'] = self.df

        
        if cat_dims is not None:
            self.cat_dims = cat_dims
            self.emb_method = emb_method
            self.emb_dims = self.emb_method(self.cat_dims)
            print(self.emb_dims)


    @classmethod
    def learn(cls, manifest_path: str): # split here and call class for each of train/val/test....?ORRR df should be df_dict

        with open(manifest_path, 'r') as f:
            mf = json.load(f)
        qoi = mf['datapackager']['qoi']
        category = qoi['category']
        continuous = qoi['continuous']
        target = qoi['target']
        informative = qoi['informative']

        ds_mf = mf['datasource']
        datasource = getattr(import_module(ds_mf['module']), ds_mf['name'])
        datasourced = datasource.init_from_dict(mf['datasource'])
        
        df = datasourced.static()
        strategy = mf['datapackager']['strategy']
        #cat_dims = {k: v for k, v in datasourced.cat_dims.items() if k in category}
        #cat_dims = {k: max(v.values()) for k, v in ds.schema['category']['mapper'].items() if k in cat_cols}
        cat_dims = [v.values() for k, v in datasourced.schema['category']['mapper'].items() if k in cat_cols]
        emb_method = emb_half_or_50

        return cls(df=df, 
                   category=category, 
                   continuous=continuous, 
                   target=target, 
                   informative=informative, 
                   strategy=strategy, 
                   cat_dims=cat_dims,
                   emb_method=emb_method)

    @classmethod
    def inference(cls, config_path: str, query: str): 

        with open(config_path, 'r') as f:
            cfg = json.load(f)

        qoi = mf['datapackager']['qoi']
        category = qoi['category']
        continuous = qoi['continuous']
        target = qoi['target']
        informative = qoi['informative']

        ds_mf = mf['datasource']
        datasource = getattr(import_module(ds_mf['module']), ds_mf['name'])
        datasourced = datasource.init_from_dict(mf['datasource'])
        
        df = datasourced.ephemeral(query)

        return cls(df=df, 
                   category=category, 
                   continuous=continuous,
                   target=target,
                   informative=informative)

    def bundle(self):
        packaged_data = {}
        for key, value in self.df_dict.items():
            packaged_data[key] = self.dataset(df=value, 
                                              category=self.category, 
                                              continuous=self.continuous,
                                              target=self.target,
                                              informative=self.informative)
        return packaged_data


        ## Class weights (for imbalances)
        #class_counts = df.category.value_counts().to_dict()
        #def sort_key(item):
        #    return self.vectorizer.category_vocab.lookup_token(item[0])
        #sorted_counts = sorted(class_counts.items(), key=sort_key)
        #frequencies = [count for _, count in sorted_counts]
        #self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)


