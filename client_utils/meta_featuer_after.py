

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller, pacf
# import re

# class Feature_Extraction:
#     def __init__(self, df, timestamp=None):
#         self.df = df
#         self.timestamp = timestamp
#         self.df[self.timestamp] = pd.to_datetime(self.df[self.timestamp])
#         self.df = self.df.sort_values(by=self.timestamp).reset_index(drop=True)
#         self.categorical_columns = []
#         self.numerical_columns = []

#     def columns_types(self):
#         from math import log
#         num_samples = len(self.df)
#         log_num_samples = log(num_samples)
#         for column in self.df.columns:
#             if column in ['Target', self.timestamp]:
#                 continue
#             try:
#                 unique_values = self.df[column].nunique()
#                 if unique_values < log_num_samples or isinstance(self.df[column].dtype, pd.CategoricalDtype):
#                     self.categorical_columns.append(column)
#                 else:
#                     self.df[column] = pd.to_numeric(self.df[column], errors='raise')
#                     self.numerical_columns.append(column)
#             except:
#                 self.categorical_columns.append(column)

#     def get_num_features(self):
#         return len(self.df.columns) - 2

#     def get_num_categorical_features(self):
#         return len(self.categorical_columns)

#     def get_num_numerical_features(self):
#         return len(self.numerical_columns)

#     def get_skewness_features(self):
#         skewness = self.df[self.numerical_columns].skew(axis=0).abs()
#         return {
#             "mean_skewness": skewness.mean(),
#             "min_skewness": skewness.min(),
#             "max_skewness": skewness.max(),
#             "std_skewness": skewness.std()
#         }

#     def get_kurtosis_features(self):
#         kurtosis = self.df[self.numerical_columns].kurtosis(axis=0).abs()
#         return {
#             "mean_kurtosis": kurtosis.mean(),
#             "min_kurtosis": kurtosis.min(),
#             "max_kurtosis": kurtosis.max(),
#             "std_kurtosis": kurtosis.std()
#         }

#     def get_symbol_counts(self):
#         symbol_stats = {}
#         if self.categorical_columns:
#             for col in self.categorical_columns:
#                 unique_counts = self.df[col].astype(str).apply(lambda x: len(set(x)))
#                 symbol_stats[col] = unique_counts
#             symbol_means = np.mean([stats.mean() for stats in symbol_stats.values()])
#             symbol_mins = np.min([stats.min() for stats in symbol_stats.values()])
#             symbol_maxs = np.max([stats.max() for stats in symbol_stats.values()])
#             symbol_stds = np.std([stats.std() for stats in symbol_stats.values()])
#         else:
#             symbol_means, symbol_mins, symbol_maxs, symbol_stds = 0, 0, 0, 0
#         return {
#             "mean_symbol_count": symbol_means,
#             "min_symbol_count": symbol_mins,
#             "max_symbol_count": symbol_maxs,
#             "std_symbol_count": symbol_stds
#         }

#     def get_significant_lags(self, target_suffix='Target'):
#         lag_pattern = re.compile(r'lag\d+_' + re.escape(target_suffix))
#         lag_columns = [col for col in self.df.columns if lag_pattern.match(col)]
#         return len(lag_columns)

#     def get_number_of_stationary_features(self, target_col):
#         stationary_count = 0
#         for col in self.df.columns:
#             if col != self.timestamp and col != target_col:
#                 series = self.df[col].dropna()
#                 if adfuller(series)[1] < 0.05:
#                     stationary_count += 1
#         return stationary_count

#     def get_number_of_stationary_features_1_dif(self, target_col):
#         stationary_count = 0
#         for col in self.df.columns:
#             if col != self.timestamp and col != target_col:
#                 series = self.df[col].dropna().diff().dropna()
#                 if adfuller(series)[1] < 0.05:
#                     stationary_count += 1
#         return stationary_count

#     def get_number_of_stationary_features_2_dif(self, target_col):
#         stationary_count = 0
#         for col in self.df.columns:
#             if col != self.timestamp and col != target_col:
#                 series = self.df[col].dropna().diff().diff().dropna()
#                 if adfuller(series)[1] < 0.05:
#                     stationary_count += 1
#         return stationary_count

#     def get_target_stationarity(self, target_col):
#         series = self.df[target_col].dropna()
#         p_value = adfuller(series)[1]
#         return p_value < 0.05

#     def get_sampling_rate(self):
#         X = pd.DataFrame({'timestamp': pd.to_datetime(self.df[self.timestamp])})
#         time_diff = X['timestamp'].diff()
#         sampling_period = time_diff.dt.total_seconds() / 3600  # sampling rate per hour
#         return sampling_period.iloc[1]

#     def get_seasonality_components(self, target_col):
#         series = self.df[target_col].dropna()
#         decomposition = sm.tsa.seasonal_decompose(series, period=1)
#         seasonal = decomposition.seasonal
#         unique_seasons = len(set(seasonal))
#         return unique_seasons

#     def get_max_seasonality_period(self, target_col, max_period=365):
#         series = self.df[target_col].dropna()
#         best_period = 1
#         max_strength = 0
#         for period in range(2, max_period + 1):
#             decomposition = sm.tsa.seasonal_decompose(series, period=period)
#             seasonal_strength = decomposition.seasonal.var()
#             if seasonal_strength > max_strength:
#                 max_strength = seasonal_strength
#                 best_period = period
#         return best_period

#     def get_min_seasonality_period(self, target_col, max_period=365):
#         series = self.df[target_col].dropna()
#         best_period = 1
#         min_strength = float('inf')
#         for period in range(2, max_period + 1):
#             decomposition = sm.tsa.seasonal_decompose(series, period=period)
#             seasonal_strength = decomposition.seasonal.var()
#             if seasonal_strength < min_strength:
#                 min_strength = seasonal_strength
#                 best_period = period
#         return best_period

#     def get_fractal_dimension_analysis(self, target_col):
#         series = self.df[target_col].dropna()
#         n = len(series)
#         max_k = int(np.log2(n))
#         k_vals = np.arange(1, max_k + 1)
#         L = np.zeros(len(k_vals))
#         for i, k in enumerate(k_vals):
#             n_k = n // k
#             Lk = np.zeros(k)
#             for m in range(k):
#                 idx = np.arange(m, n, k)
#                 Lk[m] = np.sum(np.abs(np.diff(series[idx])))
#             L[i] = np.sum(Lk) * (n - 1) / (k * n_k)
#         log_k = np.log(k_vals)
#         log_L = np.log(L)
#         coeffs = np.polyfit(log_k, log_L, 1)
#         return coeffs[0]

#     def get_insignificant_lags_between_significant(self, target_col, nlags=20):
#         series = self.df[target_col].dropna()
#         pacf_values = pacf(series, nlags=nlags)
#         conf_interval = 1.96 / np.sqrt(len(series))
#         significant_lags = [lag for lag, value in enumerate(pacf_values) if abs(value) > conf_interval]
#         if len(significant_lags) < 2:
#             return 0
#         else:
#             first_significant = significant_lags[0]
#             last_significant = significant_lags[-1]
#             insignificant_lags = sum(abs(pacf_values[lag]) < conf_interval for lag in range(first_significant + 1, last_significant))
#             return insignificant_lags

#     def extract_features(self, target_col):
#         self.columns_types()
#         skewness_features = self.get_skewness_features()
#         kurtosis_features = self.get_kurtosis_features()
#         symbol_counts = self.get_symbol_counts()
#         features = {
#             'No. Of Features': self.get_num_features(),
#             'No. Of Categorical Features': self.get_num_categorical_features(),
#             'No. Of Numerical Features': self.get_num_numerical_features(),
#             'Sampling Rate': self.get_sampling_rate(),
#             'No. Of Stationary Features': self.get_number_of_stationary_features(target_col),
#             'No. Of Stationary Features after 1st order diff': self.get_number_of_stationary_features_1_dif(target_col),
#             'No. Of Stationary Features after 2nd order diff': self.get_number_of_stationary_features_2_dif(target_col),
#             'Target Stationarity': self.get_target_stationarity(target_col),
#             'Significant Lags using pACF in Target': self.get_significant_lags(target_col),
#             'No. Of Insignificant Lags between 1st and last significant ones in Target': self.get_insignificant_lags_between_significant(target_col),
#             'No. Of Seasonality Components in Target': self.get_seasonality_components(target_col),
#             'Average Skewness of Numerical Features': skewness_features['mean_skewness'],
#             'Minimum Skewness of Numerical Features': skewness_features['min_skewness'],
#             'Maximum Skewness of Numerical Features': skewness_features['max_skewness'],
#             'Stddev Skewness of Numerical Features': skewness_features['std_skewness'],
#             'Average Kurtosis of Numerical Features': kurtosis_features['mean_kurtosis'],
#             'Minimum Kurtosis of Numerical Features': kurtosis_features['min_kurtosis'],
#             'Maximum Kurtosis of Numerical Features': kurtosis_features['max_kurtosis'],
#             'Stddev Kurtosis of Numerical Features': kurtosis_features['std_kurtosis'],
#             'Fractal Dimension Analysis of Target': self.get_fractal_dimension_analysis(target_col),
#             'Maximum Period of Seasonality Components in Target': self.get_max_seasonality_period(target_col),
#             'Minimum Period of Seasonality Components in Target': self.get_min_seasonality_period(target_col),
#             'Avg No. of Symbols per Categorical Features': symbol_counts['mean_symbol_count'],
#             'Min. No. Of Symbols per Categorical Features': symbol_counts['min_symbol_count'],
#             'Max. No. Of Symbols per Categorical Features': symbol_counts['max_symbol_count'],
#             'Stddev No. Of Symbols per categorical Features': symbol_counts['std_symbol_count']
#         }
#         return  {"meta featuers":features}

# def FEX_pipeline(df):
#     TARGET_KEYWORDS = ['Close', 'close', 'Value', 'value', 'target', 'Target']
#     TIMESTAMP_KEYWORDS = ['timestamp', 'Timestamp']
#     target_col, timestamp_col = detect_target_timestamp(df, TARGET_KEYWORDS , TIMESTAMP_KEYWORDS)
#     ext = Feature_Extraction(df, timestamp_col)
#     return ext.extract_features(target_col)

# def detect_target_timestamp(df, target_names, timestamp_names):
#     target_col = None
#     timestamp_col = None

#     for col in df.columns:
#         if col in target_names:
#             target_col = col
#         if col in timestamp_names:
#             timestamp_col = col

#     if target_col is None or timestamp_col is None:
#         raise ValueError("Couldn't find the target or timestamp columns in the DataFrame")

#     return target_col, timestamp_col

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, pacf
import re

class Feature_Extraction:
    def __init__(self, df, timestamp=None):
        self.df = df
        self.timestamp = timestamp
        self.df[self.timestamp] = pd.to_datetime(self.df[self.timestamp])
        self.df = self.df.sort_values(by=self.timestamp).reset_index(drop=True)
        self.categorical_columns = []
        self.numerical_columns = []

    def columns_types(self):
        from math import log
        num_samples = len(self.df)
        log_num_samples = log(num_samples)
        for column in self.df.columns:
            if column in ['Target', self.timestamp]:
                continue
            try:
                unique_values = self.df[column].nunique()
                if unique_values < log_num_samples or isinstance(self.df[column].dtype, pd.CategoricalDtype):
                    self.categorical_columns.append(column)
                else:
                    self.df[column] = pd.to_numeric(self.df[column], errors='raise')
                    self.numerical_columns.append(column)
            except:
                self.categorical_columns.append(column)

    def get_num_features(self):
        return len(self.df.columns) - 2

    def get_num_categorical_features(self):
        return len(self.categorical_columns)

    def get_num_numerical_features(self):
        return len(self.numerical_columns)

    def get_skewness_features(self):
        skewness = self.df[self.numerical_columns].skew(axis=0).abs()
        return {
            "mean_skewness": skewness.mean(),
            "min_skewness": skewness.min(),
            "max_skewness": skewness.max(),
            "std_skewness": skewness.std()
        }

    def get_kurtosis_features(self):
        kurtosis = self.df[self.numerical_columns].kurtosis(axis=0).abs()
        return {
            "mean_kurtosis": kurtosis.mean(),
            "min_kurtosis": kurtosis.min(),
            "max_kurtosis": kurtosis.max(),
            "std_kurtosis": kurtosis.std()
        }

    def get_symbol_counts(self):
        symbol_stats = {}
        if self.categorical_columns:
            for col in self.categorical_columns:
                unique_counts = self.df[col].astype(str).apply(lambda x: len(set(x)))
                symbol_stats[col] = unique_counts
            symbol_means = np.mean([stats.mean() for stats in symbol_stats.values()])
            symbol_mins = np.min([stats.min() for stats in symbol_stats.values()])
            symbol_maxs = np.max([stats.max() for stats in symbol_stats.values()])
            symbol_stds = np.std([stats.std() for stats in symbol_stats.values()])
        else:
            symbol_means, symbol_mins, symbol_maxs, symbol_stds = 0, 0, 0, 0
        return {
            "mean_symbol_count": symbol_means,
            "min_symbol_count": symbol_mins,
            "max_symbol_count": symbol_maxs,
            "std_symbol_count": symbol_stds
        }

    def get_significant_lags(self, target_suffix='Target'):
        lag_pattern = re.compile(r'lag\d+_' + re.escape(target_suffix))
        lag_columns = [col for col in self.df.columns if lag_pattern.match(col)]
        return len(lag_columns)

    # def get_number_of_stationary_features(self, target_col):
    #     stationary_count = 0
    #     for col in self.df.columns:
    #         if col != self.timestamp and col != target_col:
    #             series = self.df[col].dropna()
    #             if adfuller(series)[1] < 0.05:
    #                 stationary_count += 1
    #     return stationary_count
    from statsmodels.tsa.stattools import adfuller

    def get_number_of_stationary_features(self, target_col):
        stationary_count = 0
        for col in self.df.columns:
            if col != self.timestamp and col != target_col:
                series = self.df[col].dropna()
                if series.nunique() > 1:  # Ensure the series is not constant
                    if adfuller(series)[1] < 0.05:
                        stationary_count += 1
        return stationary_count


    def get_number_of_stationary_features_1_dif(self, target_col):
        stationary_count = 0
        for col in self.df.columns:
            if col != self.timestamp and col != target_col:
                series = self.df[col].dropna().diff().dropna()
                if series.nunique() > 1:
                    if adfuller(series)[1] < 0.05:
                        stationary_count += 1
        return stationary_count

    def get_number_of_stationary_features_2_dif(self, target_col):
        stationary_count = 0
        for col in self.df.columns:
            if col != self.timestamp and col != target_col:
                series = self.df[col].dropna().diff().diff().dropna()
                if series.nunique() > 1:
                    if adfuller(series)[1] < 0.05:
                        stationary_count += 1
        return stationary_count

    def get_target_stationarity(self, target_col):
        series = self.df[target_col].dropna()
        p_value = adfuller(series)[1]
        return p_value < 0.05

    def get_sampling_rate(self):
        X = pd.DataFrame({'timestamp': pd.to_datetime(self.df[self.timestamp])})
        time_diff = X['timestamp'].diff()
        sampling_period = time_diff.dt.total_seconds() / 3600  # sampling rate per hour
        return sampling_period.iloc[1]

    def get_seasonality_components(self, target_col):
        series = self.df[target_col].dropna()
        try:
            decomposition = sm.tsa.seasonal_decompose(series, period=1)
            seasonal = decomposition.seasonal
            unique_seasons = len(set(seasonal))
        except ValueError:
            unique_seasons = 0
        return unique_seasons

    def get_max_seasonality_period(self, target_col, max_period=365):
        series = self.df[target_col].dropna()
        best_period = 1
        max_strength = 0
        try:
            for period in range(2, max_period + 1):
                decomposition = sm.tsa.seasonal_decompose(series, period=period)
                seasonal_strength = decomposition.seasonal.var()
                if seasonal_strength > max_strength:
                    max_strength = seasonal_strength
                    best_period = period
        except ValueError:
            pass
        return best_period

    def get_min_seasonality_period(self, target_col, max_period=365):
        series = self.df[target_col].dropna()
        best_period = 1
        min_strength = float('inf')
        try:
            for period in range(2, max_period + 1):
                decomposition = sm.tsa.seasonal_decompose(series, period=period)
                seasonal_strength = decomposition.seasonal.var()
                if seasonal_strength < min_strength:
                    min_strength = seasonal_strength
                    best_period = period
        except ValueError:
            pass
        return best_period

    def get_fractal_dimension_analysis(self, target_col):
        series = self.df[target_col].dropna()
        n = len(series)
        max_k = int(np.log2(n))
        k_vals = np.arange(1, max_k + 1)
        L = np.zeros(len(k_vals))
        for i, k in enumerate(k_vals):
            n_k = n // k
            Lk = np.zeros(k)
            for m in range(k):
                idx = np.arange(m, n, k)
                Lk[m] = np.sum(np.abs(np.diff(series[idx])))
            L[i] = np.sum(Lk) * (n - 1) / (k * n_k)
        log_k = np.log(k_vals)
        log_L = np.log(L)
        coeffs = np.polyfit(log_k, log_L, 1)
        return coeffs[0]

    def get_insignificant_lags_between_significant(self, target_col, nlags=20):
        series = self.df[target_col].dropna()
        pacf_values = pacf(series, nlags=nlags)
        conf_interval = 1.96 / np.sqrt(len(series))
        significant_lags = [lag for lag, value in enumerate(pacf_values) if abs(value) > conf_interval]
        if len(significant_lags) < 2:
            return 0
        else:
            first_significant = significant_lags[0]
            last_significant = significant_lags[-1]
            insignificant_lags = sum(abs(pacf_values[lag]) < conf_interval for lag in range(first_significant + 1, last_significant))
            return insignificant_lags

    def extract_features(self, target_col):
        self.columns_types()
        skewness_features = self.get_skewness_features()
        kurtosis_features = self.get_kurtosis_features()
        symbol_counts = self.get_symbol_counts()
        features = {
            'No. Of Features': self.get_num_features(),
            'No. Of Categorical Features': self.get_num_categorical_features(),
            'No. Of Numerical Features': self.get_num_numerical_features(),
            'Sampling Rate': self.get_sampling_rate(),
            'No. Of Stationary Features': self.get_number_of_stationary_features(target_col),
            'No. Of Stationary Features after 1st order diff': self.get_number_of_stationary_features_1_dif(target_col),
            'No. Of Stationary Features after 2nd order diff': self.get_number_of_stationary_features_2_dif(target_col),
            'Target Stationarity': self.get_target_stationarity(target_col),
            'Significant Lags using pACF in Target': self.get_significant_lags(target_col),
            'No. Of Insignificant Lags between 1st and last significant ones in Target': self.get_insignificant_lags_between_significant(target_col),
            'No. Of Seasonality Components in Target': self.get_seasonality_components(target_col),
            'Average Skewness of Numerical Features': skewness_features['mean_skewness'],
            'Minimum Skewness of Numerical Features': skewness_features['min_skewness'],
            'Maximum Skewness of Numerical Features': skewness_features['max_skewness'],
            'Stddev Skewness of Numerical Features': skewness_features['std_skewness'],
            'Average Kurtosis of Numerical Features': kurtosis_features['mean_kurtosis'],
            'Minimum Kurtosis of Numerical Features': kurtosis_features['min_kurtosis'],
            'Maximum Kurtosis of Numerical Features': kurtosis_features['max_kurtosis'],
            'Stddev Kurtosis of Numerical Features': kurtosis_features['std_kurtosis'],
            'Fractal Dimension Analysis of Target': self.get_fractal_dimension_analysis(target_col),
            'Maximum Period of Seasonality Components in Target': self.get_max_seasonality_period(target_col),
            'Minimum Period of Seasonality Components in Target': self.get_min_seasonality_period(target_col),
            'Avg No. of Symbols per Categorical Features': symbol_counts['mean_symbol_count'],
            'Min. No. Of Symbols per Categorical Features': symbol_counts['min_symbol_count'],
            'Max. No. Of Symbols per Categorical Features': symbol_counts['max_symbol_count'],
            'Stddev No. Of Symbols per categorical Features': symbol_counts['std_symbol_count']
        }
        return  {"meta featuers":features}

def FEX_pipeline(df):
    TARGET_KEYWORDS = ['Close', 'close', 'Value', 'value', 'target', 'Target']
    TIMESTAMP_KEYWORDS = ['timestamp', 'Timestamp']
    target_col, timestamp_col = detect_target_timestamp(df, TARGET_KEYWORDS , TIMESTAMP_KEYWORDS)
    ext = Feature_Extraction(df, timestamp_col)
    return ext.extract_features(target_col)

def detect_target_timestamp(df, target_names, timestamp_names):
    target_col = None
    timestamp_col = None

    for col in df.columns:
        if col in target_names:
            target_col = col
        if col in timestamp_names:
            timestamp_col = col

    if target_col is None or timestamp_col is None:
        raise ValueError("Couldn't find the target or timestamp columns in the DataFrame")

    return target_col, timestamp_col
