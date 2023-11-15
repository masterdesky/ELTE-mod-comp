import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup

from .jdcal import jd2g

def fetch_data(p_start=1, p_end=15):
    '''
    Fetches observation data for Betelgeuse from the AAVSO website.

    Parameters
    ----------
    first_page : int
        The first page number to start fetching data from.
    last_page : int
        The last page number to fetch data up to.

    Returns
    -------
    tuple: Two arrays containing Julian Dates and magnitudes.
    '''
    jd = []
    mag = []
    url_base = 'https://www.aavso.org/apps/webobs/results/?star=betelgeuse&num_results=200&obs_types=vis&page={}'

    for p_i in range(p_start, p_end + 1):
        r = requests.get(url_base.format(p_i))
        soup = BeautifulSoup(markup=r.content, features='html.parser')

        # The rows are contained in <tr> elements inside a singular <tbody> element
        table = soup.select('tbody tr')

        for row in table[::4]:  # Relevant data is in every 4th <tr> tag
            # Extract the Julian Date (3rd column) and magnitude (5th column)
            jd_i = row.select('td')[2].get_text()
            mag_i = row.select('td')[4].get_text()
            try:
                # Test if magnitude value is float or not
                _ = float(mag_i)
                jd.append(jd_i)
                mag.append(mag_i)
            except:
                continue

    jd = np.array(jd, dtype=np.float64)
    mag = np.array(mag, dtype=np.float64)

    return jd, mag


def aggregate_data(jd, mag):
    '''
    Aggregate data and calculate average magnitude and standard error
    for each day.
    '''
    gd = jd2g(jd)          # Gregorian dates
    jd_i = np.int64(jd)    # Julian Day Number
    ujd = np.unique(jd_i)  # Julian Days with observations
    ugd = jd2g(ujd)        # Gregorian Days with observations

    # Calculate daily averages and errors for V-band magnitudes
    df = pd.DataFrame(data={'jd_i': jd_i, 'mag': mag})
    group = df['mag'].groupby(jd_i)
    mag, err = group.mean(), group.std()

    # Calculate relative apparent V-band magnitude compared to the
    # average of 0.5
    rel_L = np.power(2.512, mag-0.5) * 100

    data = pd.DataFrame(data={
        'gd': ugd, 'mag': mag, 'err': err, 'rel_L': rel_L
    })
    return data


def filter_data(data, min_date, max_date, max_std):
    # Drop missing values
    data.dropna(axis=0, how='any', inplace=True)
    # Filter by date
    data = data[(data['gd'] > min_date) & (data['gd'] < max_date)]
    # Filter by error
    data = data[data['err'] < max_std]
    return data