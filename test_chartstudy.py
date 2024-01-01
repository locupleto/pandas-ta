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
import numpy as np
from marketdata.database import Database
from pandas_ta.volume.avwap import pivot, anchored_vwap, rolling_standardize

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

    def _test_rwd(self):

        df = self.df
        print(df.index)

        df = df.ta.rwd(high = df['high'], 
                        low = df['low'], 
                        close = df['close'],
                        smooth_type='ezls',
                        smooth_length=6,
                        smooth_gain=3,
                        probability_output=False) 
        print(df.columns)

        # Save to CSV
        df.to_csv('/Users/urban/Desktop/port_data.csv', index=False)

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

    def _test_avwap(self):
        # Load your stock data into a DataFrame
        df = self.df.tail(200)
        left_strength = 10
        right_strength = 5

        # Apply the avwap function
        df = df.ta.avwap(high=df['high'], low=df['low'], close=df['close'],
                        volume=df['volume'], left_strength=left_strength,
                        right_strength=right_strength, bands=None)

        # Identify the pivot high and low columns
        pivot_high_col = f"PIVOT_HIGH_{left_strength}_{right_strength}"
        pivot_low_col = f"PIVOT_LOW_{left_strength}_{right_strength}"

        # Plotting the stock chart
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close', color='blue')

        # Track the index to start the next segment
        next_start_high = next_start_low = 0

        # Iterate over DataFrame rows to plot segments and pivots
        for i in range(len(df)):
            # Handle AVWAP_HIGH segments and pivot markers
            if df.iloc[i][pivot_high_col]:
                if next_start_high < i:
                    plt.plot(df.iloc[next_start_high:i].index, df['AVWAP_HIGH'][next_start_high:i], color='red')
                next_start_high = i  # Start of next segment
                plt.scatter(df.index[i], df.iloc[i]['high'], color='red', marker='x')  # Pivot marker

            # Handle AVWAP_LOW segments and pivot markers
            if df.iloc[i][pivot_low_col]:
                if next_start_low < i:
                    plt.plot(df.iloc[next_start_low:i].index, df['AVWAP_LOW'][next_start_low:i], color='green')
                next_start_low = i  # Start of next segment
                plt.scatter(df.index[i], df.iloc[i]['low'], color='green', marker='x')  # Pivot marker

        # Plot the remaining parts of AVWAP_HIGH and AVWAP_LOW, if any
        if next_start_high < len(df):
            plt.plot(df.iloc[next_start_high:].index, df['AVWAP_HIGH'][next_start_high:], color='red', label='AVWAP High')
        if next_start_low < len(df):
            plt.plot(df.iloc[next_start_low:].index, df['AVWAP_LOW'][next_start_low:], color='green', label='AVWAP Low')

        plt.title('Stock Chart with Segmented AVWAP High and Low')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the DataFrame to a CSV file
        df.to_csv('/Users/urban/Desktop/test_avwap.csv')

    def test_avwap_osc(self):
        # Load your stock data into a DataFrame
        df = self.df.tail(200)
        left_strength = 5
        right_strength = 5
        window = 50

        # Apply the avwap function
        df = df.ta.avwap(high=df['high'], low=df['low'], close=df['close'],
                        volume=df['volume'], left_strength=left_strength,
                        right_strength=right_strength, bands=None)

        # Extract indices of pivot points
        pivot_high_indices = df.index[df[f"PIVOT_HIGH_{left_strength}_{right_strength}"]].tolist()
        pivot_low_indices = df.index[df[f"PIVOT_LOW_{left_strength}_{right_strength}"]].tolist()

        # Initialize AVWAP oscillator values
        avwap_low_values = np.zeros(len(df))
        avwap_high_values = np.zeros(len(df))

        # Calculate oscillator values
        for index, row in df.iterrows():
            price = row['close']

            # Calculate AVWAP_HIGH oscillator
            if index in pivot_high_indices:
                anchored_vwap_values_high = row['AVWAP_HIGH']
            if pivot_high_indices and index >= min(pivot_high_indices):
                avwap_high_values[df.index.get_loc(index)] = anchored_vwap_values_high - price 

            # Calculate AVWAP_LOW oscillator
            if index in pivot_low_indices:
                anchored_vwap_values_low = row['AVWAP_LOW']
            if pivot_low_indices and index >= min(pivot_low_indices):
                avwap_low_values[df.index.get_loc(index)] =  anchored_vwap_values_low - price 

        # Calculate relative oscillator values
        df['AVWAP_LOW_Oscillator'] = ((df['AVWAP_LOW'] - df['close']) / df['close']).astype(float)
        df['AVWAP_HIGH_Oscillator'] = ((df['AVWAP_HIGH'] - df['close']) / df['close']).astype(float)

        # Normalize the oscillator values
        df['AVWAP_LOW_Oscillator_Norm'] = rolling_standardize(df['AVWAP_LOW_Oscillator'], window)
        df['AVWAP_HIGH_Oscillator_Norm'] = rolling_standardize(df['AVWAP_HIGH_Oscillator'], window)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Close', color='blue')
        plt.title('Stock Chart with AVWAP')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['AVWAP_LOW_Oscillator_Norm'], label='AVWAP Low Oscillator', color='green')
        plt.plot(df.index, df['AVWAP_HIGH_Oscillator_Norm'], label='AVWAP High Oscillator', color='red')
        plt.title('AVWAP Oscillator')
        plt.xlabel('Date')
        plt.ylabel('Oscillator Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        stats = df[['AVWAP_LOW_Oscillator_Norm', 'AVWAP_HIGH_Oscillator_Norm']].describe()
        print(stats)

        # Save the DataFrame to a CSV file
        df.to_csv('/Users/urban/Desktop/test_avwap_osc.csv')


if __name__ == '__main__':
    unittest.main()
