import pandas as pd
from math import log
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CustomLabelEncoder(LabelEncoder):
    def transform(self, y):
        # Original classes plus a placeholder for unseen
        seen_classes = np.append(self.classes_, '<UNK>')
        unseen_label = np.where(seen_classes == '<UNK>')[0][0]

        # Handle unseen labels
        return np.array(
            [np.where(seen_classes == label)[0][0] if label in self.classes_ else unseen_label for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class ReadPreprocessData:
    def __init__(self):
        self.columns_types = {'categorical': [], 'numerical': []}
        self.categorical_columns = None
        self.numerical_columns = None
        self.dataset_type = None
        self.label_encoders = {}

    def fit_transform(self, X):
        self.data = X.copy()
        self.data = self.data.sort_values(by='Timestamp')
        self.detect_columns_types()
        self.detect_dataset_type()
        self.categorical_columns = self.columns_types['categorical']
        self.numerical_columns = self.columns_types['numerical']
        self.encode_categorical(fit=True)
        self.fill_missing()
        self.drop_rows_with_nans()
        self.drop_columns_with_nans()
        return self.data, self.columns_types, self.dataset_type

    def transform(self, X):
        self.data = X.copy()
        self.data = self.data.sort_values(by='Timestamp')
        self.encode_categorical(fit=False)
        self.fill_missing()
        self.drop_rows_with_nans()
        self.drop_columns_with_nans()
        return self.data

    def encode_categorical(self, fit=True):
        for col in self.categorical_columns:
            if fit:
                le = CustomLabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    self.data[col] = le.transform(self.data[col].astype(str))
                else:
                    raise ValueError(f"LabelEncoder not found for column: {col}")

    # def fill_missing(self):
    #     for col in self.categorical_columns:
    #         self.data[col] = self.data[col].ffill()

    #     for col in self.numerical_columns + [self.columns_types["target"]]:
            
    #        if self.data.iloc[0].isnull().any():
    #                 self.data= self.data.iloc[1:]  
    #        self.data[col] = self.data[col].interpolate(method='linear')
        
    # def fill_missing(self):
    # # Check if the first row has any missing values and drop it until it doesn't
    #     while self.data.iloc[0].isnull().any():
    #         self.data = self.data.iloc[1:].reset_index(drop=True)

    #     # Forward fill for categorical columns
    #     for col in self.categorical_columns:
    #         self.data[col] = self.data[col].ffill()

    #     # Linear interpolation for numerical columns and target column
    #     for col in self.numerical_columns + [self.columns_types["target"]]:
    #         self.data[col] = self.data[col].interpolate(method='linear')

    #     # Check if there are still missing values at the beginning after interpolation
    #     while self.data.iloc[0].isnull().any():
    #         self.data = self.data.iloc[1:].reset_index(drop=True)
    def fill_missing(self):
    # Check if the DataFrame is empty
        if self.data.empty:
            print("DataFrame is empty.")
            return

        # Drop rows with missing values at the beginning until the first row has no missing values
        while not self.data.empty and self.data.iloc[0].isnull().any():
            self.data = self.data.iloc[1:].reset_index(drop=True)

        # If the DataFrame is empty after dropping rows, return
        if self.data.empty:
            print("DataFrame became empty after dropping rows with missing values.")
            return

        # Forward fill for categorical columns
        for col in self.categorical_columns:
            self.data[col] = self.data[col].ffill()

        # Linear interpolation for numerical columns and target column
        for col in self.numerical_columns + [self.columns_types["target"]]:
            self.data[col] = self.data[col].interpolate(method='linear')

        # Drop rows with missing values at the beginning again after interpolation
        while not self.data.empty and self.data.iloc[0].isnull().any():
            self.data = self.data.iloc[1:].reset_index(drop=True)
        
        # If the DataFrame is empty after the second drop, print a message
        if self.data.empty:
            print("DataFrame became empty after the second drop of rows with missing values.")


    # def fill_missing(self):
    #     if self.dataset_type == "univariate":
    #         self.data = self.data.interpolate(method='linear')
    #     else:
    #         # Apply linear interpolation for numerical columns
    #         num_cols = self.data.select_dtypes(include=['number']).columns
    #         self.data[num_cols] = self.data[num_cols].interpolate(method='linear')
            
    #         # Forward fill for categorical columns
    #         cat_cols = self.data.select_dtypes(include=['object']).columns
    #         self.data[cat_cols] = self.data[cat_cols].ffill()

    def detect_columns_types(self):
        num_samples = len(self.data)
        log_num_samples = log(num_samples)

        for column in self.data.columns:
            if column == 'Target':
                self.columns_types['target'] = column
            elif column == 'Timestamp':
                self.columns_types['timestamp'] = column
            else:
                unique_values = self.data[column].nunique()
                if unique_values < log_num_samples or isinstance(self.data[column].dtype, pd.CategoricalDtype):
                    self.columns_types['categorical'].append(column)
                else:
                    self.columns_types['numerical'].append(column)

    def detect_dataset_type(self):
        if self.columns_types['categorical'] or self.columns_types['numerical']:
            self.dataset_type = "multivariate"
        else:
            self.dataset_type = "univariate"
    def drop_rows_with_nans(self):
    
        # Replace infinite values with NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        nan_threshold = 5  # Example threshold: Rows with 5 or more NaN values will be included

        # Filter rows with more than 'nan_threshold' NaN values
        rows_to_drop = self.data[self.data.isna().sum(axis=1) >= nan_threshold].index

        
        # Drop rows with excessive NaN values
        self.data.drop(index=rows_to_drop, inplace=True)

    def drop_columns_with_nans(self, threshold=0.7):
        # Replace infinite values with NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Calculate proportion of NaN values in each column
        nan_proportion = self.data.isna().mean()
        
        # Filter columns where the proportion of NaN values exceeds the threshold
        columns_to_drop = nan_proportion[nan_proportion > threshold].index
        
        # Drop columns with excessive NaN values
        self.data.drop(columns=columns_to_drop, inplace=True)    
        


# data = pd.read_csv(r'D:\OneDrive\Desktop\fedrated_ver3 - Copy\Data\1004_split_1.csv')
# def check_data_health(data):
#         has_inf = data.isin([np.inf, -np.inf]).any().any()
#         has_nan = data.isna().any().any()
#         return {'has_inf': has_inf, 'has_nan': has_nan}
# def check_columns_name(data):
#         # Check the Target name
#         target_keywords = ['Close', 'close', 'value', 'Value']
#         for col in data.columns:
#             if any(keyword in col for keyword in target_keywords):
#                 data.rename(columns={col: 'Target'}, inplace=True)
#                 break

#         # Check the Timestamp name
#         timestamp_keywords = ['timestamp', 'Timestamp']
#         for col in data.columns:
#             if any(keyword in col for keyword in timestamp_keywords):
#                 data.rename(columns={col: 'Timestamp'}, inplace=True)
#                 break
# preprocessor = ReadPreprocessData()
# check_columns_name(data)

# processed_data, columns_types, dataset_type = preprocessor.fit_transform(data)
# rows_with_nan = processed_data[processed_data.isna().any(axis=1)]
# print("rows with nan",rows_with_nan)
# # processed_data["Target"] = processed_data["Target"].interpolate(method='linear')
# print(check_data_health(processed_data))
# print("Processed Data:")
# print(processed_data)
# print("Columns Types:", columns_types)
# print("Dataset Type:", dataset_type)
