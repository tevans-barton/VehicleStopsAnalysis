#!/usr/bin/env python

import numpy as np
import pandas as pd

def clean_stops(stops_list):
    """
    cleans and concatenates a list of
    dataframes (each a year of SD police 
    vehicle stops data) into a single
    dataframe.

    NOTE: this is the function used to create
    the data included in the project! you do NOT
    need to use this function. It is for 
    your reference.
    """

    stops = pd.concat(stops_list)
    # service area: should be clean other than 'Unkown'
    stops['service_area'] = stops.service_area.replace({'Unknown': np.NaN})
    
    # rename index to remove duplicate stop_ids
    # note that this column is necessary to join to the search details
    # data we don't use here
    stops['stop_id'] = np.arange(len(stops))

    # Remove malformed ages and ages that are too young/old
    # this is < 0.001 of data
    stops['subject_age'] = pd.to_numeric(stops['subject_age'], errors='coerce')
    stops['subject_age'] = stops.subject_age.apply(
        lambda x: np.NaN if ((float(x) < 15) | (float(x) > 99)) else x)

    # Null out malformed time stamps
    stops['timestamp'] = pd.to_datetime(stops['timestamp'], errors='coerce')

    # Clean Y/N fields
    stops['searched'] = stops.searched.apply(clean_binary_cols)
    stops['sd_resident'] = stops.sd_resident.apply(clean_binary_cols)

    # remove 2018 (incomplete) and
    # and remove malformed times (this is < 0.001 of the data)
    stops = stops.dropna(subset=['timestamp']).loc[stops.timestamp < '2018']

    # remove columns we won't be using
    stops = stops.drop(
        ['stop_date', 'stop_time', 'obtained_consent', 
         'contraband_found', 'property_seized', 'arrested'], 
        axis=1
    )

    return stops


def clean_binary_cols(x):
    """
    Cleans Y/N columns to 0/1 columns
    """
    if x in ['Y', 'y']:
        return 1
    elif x in ['N', 'n']:
        return 0
    else:
        return np.NaN


def impute_target(stops):
    """
    imputes the search column in the
    stops dataset by drawing the from distribution
    of searches conditional on the service_area.
    """

    def impute(searched):
        p = searched.mean()
        return searched.apply(
            lambda x: np.random.choice([0, 1], p=[1 - p, p]) if pd.isnull(x) else x
        )

    searched = (
        stops
        .fillna({'service_area': 'NULL'})
        .groupby('service_area')
        .searched
        .apply(impute)
    )

    return stops.assign(searched=searched)
