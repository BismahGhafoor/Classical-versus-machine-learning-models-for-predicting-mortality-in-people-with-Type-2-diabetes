# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:57:07 2022

@author: ss1279
"""
import pandas as pd
import numpy as np
import scipy.stats
# import os
# from tableone import TableOne
# import matplotlib.pyplot as plt
# import time
# import glob

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# =============================================================================
#
# =============================================================================
def remap_eth(df,col):
    dict_labels = {col:{
        'Indian':'South Asian',
        'Pakistani':'South Asian',
        'Bangladesi':'South Asian',
        'Bl_Carib':'Black',
        'Bl_Afric':'Black',
        'Bl_Other':'Black',
        'Other':'Mixed/Other',
        'Mixed':'Mixed/Other',
        'Chinese' :'Mixed/Other',
        'Oth_Asian' :'Mixed/Other'
        }}
    for field, values in dict_labels.items():
        df.replace({field:values},inplace=True)
    return df

def nperc_counts(df, col):
    a = df[col].value_counts(dropna=False)
    b = df[col].value_counts(normalize=True,dropna=False).mul(100).round(4).astype(str) + '%'
    print(pd.DataFrame(dict(count=a, perc=b)).rename_axis(col).reset_index())


def calc_gfr(df):
    """
    Calculates the estimated Glomerular Filteration Rate(eGFR)
    Based on CKD-EPI equation.
    eGFR = 141 x min(SCr/κ, 1)^α x max(SCr /κ, 1)^-1.209 x
    0.993^Age x 1.018 [if female] x 1.159 [if Black]

    Arguments:
        eGFR (estimated glomerular filtration rate) = mL/min/1.73 m2
        SCr (standardized serum creatinine) = mg/dL
        κ = 0.7 (females) or 0.9 (males)
        α = -0.329 (females) or -0.411 (males)
        min = indicates the minimum of SCr/κ or 1
        max = indicates the maximum of SCr/κ or 1
        age = years
    Returns:
        gfr: Patient's eGFR in mL/min rounded to decimal places.
    """
    df["kappa"] = 0.9  # Male
    df["alpha"] = -0.411  # Male
    df["constant1"] = 1  # Male
    df["constant2"] = 1  # Non- Black
    #To convert μmol/l to mg/dl, multiply by 0.0113

    df.loc[(df["gender"] == 1), "kappa"] = 0.7  # Female
    df.loc[(df["gender"] == 1), "alpha"] = -0.329  # Female
    df.loc[(df["gender"] == 1), "constant1"] = 1.018  # Female
    df.loc[(df["ethnicity"] == "Black"), "constant2"] = 1.159

    df["egfr"] = (
        141
        * (np.minimum((df["serumc_mgdL"] / df["kappa"]), 1) ** df["alpha"])
        * (np.maximum((df["serumc_mgdL"] / df["kappa"]), 1) ** (-1.209))
        * (0.993 ** df["age"])
        * df["constant1"]
        * df["constant2"]
    )

    df.drop(["kappa", "alpha", "constant1", "constant2", "age"], axis=1, inplace=True)
    return df



def save_long_format_data(df, save_long_format, name):
    if (save_long_format==True):
        df.to_csv(
            f"8.Flat_files/{name}_longitudinal_data.txt",
            sep="\t",
            index=False,
            date_format="%d/%m/%Y",
        )

def read_long_format_data(name):
    df = pd.read_csv(
        f'8.Flat_files/risk_factors_longitudinal_data/{name}_longitudinal_data.txt',
        sep='\t',
        parse_dates = ['eventdate','indexdate'],
        dayfirst = True,
        header=0
        )
    return df

def lcf(data):
    confidence=0.95
    n = data.count()
    m, se = data.mean(), data.sem() #scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h


def ucf(data):
    confidence=0.95
    n = data.count()
    m, se = data.mean(), data.sem() #scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m+h

def perc(data):
    a = data.size
    b = data.count()
    #print (a, b)
    perc = b*100/a
    return perc


# dates= pd.date_range('2017-04-01','2022-05-01' , freq='1M')-pd.offsets.MonthEnd(1)
# datestr = pd.date_range('2017-03-01','2022-03-01',
#               freq='MS').strftime("%Y-%b").tolist()

# #t2dm[datestr] = pd.DataFrame([[0]*len(datestr)], index=t2dm.index)
# t2dm[datestr] = pd.DataFrame([dates.tolist()], index=t2dm.index)

# for (_, colname) in enumerate(t2dm.iloc[:,3:]):
#     #print(index, colname, t2dm[colname])#.values)
#     t2dm[colname] = np.where(
#         (t2dm.cond_t2dm <= t2dm[colname]),1,np.nan,
#     )
# del dates, datestr, colname
# t2dm = pd.wide_to_long(t2dm,
#                        stubnames=["2017", "2018","2019",
#                         "2020","2021","2022"],
#                        i=["patid",'cond_t2dm'], j="year",
#                        sep='-', suffix=r'\w+')




# t2dm = patient[patient.cond_t2dm.notnull()]
# t2dm.count()
# t2dm.columns
# t2dm = t2dm[['patid', 'cond_t2dm']]
# t2dm['cond_t2dm'].describe()
# t2dm = t2dm.sort_values(['cond_t2dm', 'patid']).reset_index(drop=True)
# t2dm['startdate'] = pd.Timestamp('2017-03-01')
# t2dm['enddate'] = pd.Timestamp('2022-03-01')

# t2dm = t2dm.assign(
#     Date=lambda dfa: dfa.apply(
#         lambda r: pd.date_range(r["startdate"], r["enddate"],
#         freq='1M'), axis=1)).explode("Date")

# t2dm['months'] = pd.to_datetime(t2dm['Date'],
#                                  format='%m%Y',
#                                  errors='coerce').dt.to_period('m')

# t2dm['monthly_cohort'] = np.where(
#         (t2dm.cond_t2dm <= t2dm.Date),1,0,#np.nan,
#     )
# a=t2dm.iloc[4500000:4700000,:]

# t2dm.groupby(["months"])['monthly_cohort'].sum().reset_index(
#         name="count"
#     ).sort_values(["count"])
# t2dm.columns
# t2dm = t2dm[['patid', 'cond_t2dm','months','monthly_cohort']]
# #t2dm = t2dm[t2dm.monthly_cohort.notnull()]
# t2dm = t2dm.sort_values(['patid', 'months']).reset_index(drop=True)

# t2dm_bp = bp[bp.patid.isin(t2dm.patid)]
# t2dm_bp = t2dm_bp.sort_values(['patid', 'eventdate']).reset_index(drop=True)
# t2dm_bp = t2dm_bp[(t2dm_bp.eventdate >= pd.Timestamp('2017-03-01'))
#                   & (t2dm_bp.eventdate <= pd.Timestamp('2022-03-31'))]
# t2dm_bp['months'] = pd.to_datetime(t2dm_bp['eventdate'],
#                                  format='%m%Y',
#                                  errors='coerce').dt.to_period('m')
# t2dm_bp = t2dm_bp[['patid', 'diastolic', 'systolic', 'months']]
# t2dm_bp = t2dm_bp.sample(frac=1).drop_duplicates(
#         subset=['patid', 'months'], keep='last'
#         )
# t2dm_bp = t2dm_bp.sort_values(['patid', 'months']).reset_index(drop=True)


# t2dm = t2dm.merge(t2dm_bp, on=["patid", "months"], how="left")












