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
        pass

    @classmethod
    def tearDownClass(cls):
        # This will run once after all tests
        cls.db.close()
        pass

    def test_1(self):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        timeframe = "1d"

if __name__ == '__main__':
    unittest.main()
