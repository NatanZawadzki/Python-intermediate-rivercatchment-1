"""Module containing models representing catchment data.

The Model layer is responsible for the 'business logic' part of the software.

Catchment data is held in a Pandas dataframe (2D array) where each column contains
data for a single measurement site, and each row represents a single measurement
time across all sites.
"""

import pandas as pd
import numpy as np
from functools import reduce, wraps

def num_data_above_threshold(data,site_id,threshold):
    '''


    '''
    def count_above_threshold(a,b):
        if b:
            return a + 1
        else:
            return a

    above_threshold = map(lambda x: x>threshold, data[site_id])
    return reduce(count_above_threshold,above_threshold,0)
def daily_above_threshold(data, site_id, threshold): -> bool
    '''
    data: Pandas dataframe with column named site_id
    site_id: string column name 
    threshold: numerical 
    '''
    return list(map(lambda x: x > threshold, data[site_id]))


def data_normalise(data):
    max = np.array(np.max(data, axis=0))
    return data / max[np.newaxis,:]

def read_variable_from_csv(filename):
    """Reads a named variable from a CSV file, and returns a
    pandas dataframe containing that variable. The CSV file must contain
    a column of dates, a column of site ID's, and (one or more) columns
    of data - only one of which will be read.

    :param filename: Filename of CSV to load
    :return: 2D array of given variable. Index will be dates,
             Columns will be the individual sites
    """
    dataset = pd.read_csv(filename, usecols=['Date', 'Site', 'Rainfall (mm)'])

    dataset = dataset.rename({'Date':'OldDate'}, axis='columns')
    dataset['Date'] = [pd.to_datetime(x,dayfirst=True) for x in dataset['OldDate']]
    dataset = dataset.drop('OldDate', axis='columns')

    newdataset = pd.DataFrame(index=dataset['Date'].unique())

    for site in dataset['Site'].unique():
        newdataset[site] = dataset[dataset['Site'] == site].set_index('Date')["Rainfall (mm)"]

    newdataset = newdataset.sort_index()

    return newdataset

def daily_total(data):
    """Calculate the daily total of a 2d data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).sum()

def daily_mean(data):
    """Calculate the daily mean of a 2d data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).mean()


def daily_max(data):
    """Calculate the daily max of a 2d data array.
    Index must be np.datetime64 compatible format."""
    return data.groupby(data.index.date).max()


def daily_min(data):
    """Calculate the daily min of a 2d data array.
    Index must be np.datetime64 compatible format.

    :param data: 2D array
    :returns: minimum of 2D array by day (array)
    """
    return data.groupby(data.index.date).min()

