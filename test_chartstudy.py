# ============================================================================
# -*- coding: utf-8 -*-
#
# Package : tests
# Module  : test_chartstudy.py
#
# Description: test cases for the chartstudy module based on pandas_ta 
# indicators.    
#
# Dependencies: git+https://github.com/locupleto/marketdata-db
#               finsymbols, eod, pandas, pandas_ta       
#
# History       Rev   Description
# 2023-11-11    urot  Created 
# ============================================================================

import unittest
import logging
import sys

# pip install git+https://github.com/locupleto/marketdata-db
# pip install finsymbols
# pip install eod
# pip install matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from marketdata.database import Database

# pip install pandas
import pandas as pd  

# pip install pandas_ta
import pandas_ta as ta 

class MyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls): 
        # Configure the logging module to write log messages to the console
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

        # Open database once before running all test cases
        cls.eod_db_file = '/Volumes/Work/marketdata/marketdata-test.db' 
        cls.license_file = '/Users/urban/api-keys/API_KEY_EODHD.txt'
        cls.db = Database()
        cls.db.open(cls.eod_db_file, cls.license_file)

        cls.name, cls.daily_df = cls.db.get_named_eod_data('AAPL', 'US')
        #cls.name, cls.daily_df = cls.db.get_named_eod_data('ATCO-B', 'ST')
        cls.weekly_df = cls.db.daily_to_weekly(cls.daily_df)
        cls.monthly_df = cls.db.daily_to_monthly(cls.daily_df)
        cls.yearly_df = cls.db.daily_to_yearly(cls.daily_df)
        cls.df = cls.daily_df

    @classmethod
    def tearDownClass(cls):
        # This will run once after all tests
        cls.db.close()

    def test_1(self):

        df = self.df
        df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)

        df = df.ta.rwd(high = df['high'], 
                        low = df['low'], 
                        close = df['close'],
                        smooth_type='ezls',
                        smooth_length=6,
                        smooth_gain=3,
                        probability_output=False) 
        print(df.columns)

        # Now, select the last 200 rows of the specific column
        data = df['RWD_PEAK_8-65_252_ezls_6_3'].tail(120)

        # Plotting as a time series
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data, color='blue', linestyle='-')
        plt.title('Smoothed Random Walk Peak Value Over Time (Last 200 Bars)')
        plt.xlabel('Date')
        plt.ylabel('Smoothed Random Walk Peak Value')
        plt.grid(True)
        plt.show()

    def xtest_1(self):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        timeframe = "1d"

if __name__ == '__main__':
    unittest.main()
