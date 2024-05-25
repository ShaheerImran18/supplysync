import os
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from datetime import datetime
from tensorflow.keras.regularizers import L1
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error
from odoo import api, fields, models

start_ye = None
end_ye = None


class Sku(models.Model):
    _name = "supplysync.sku"
    _description = "SupplySync SKU"

    # product input fields
    sku = fields.Char("SKU")
    type = fields.Char("Type")
    category_L1 = fields.Char("Category_L1")
    category_L2 = fields.Char("Category_L2")
    vendor = fields.Char("Vendor")
    foreQuant = fields.Integer("Forecasted Quantity", default=False)
    forePeriod = fields.Date("Forecast Period", default=False)
    rank = fields.Integer("Rank", default=False)

    @api.model
    def run_model(self):
        # check if configs are set
        if start_ye is None or end_ye is None:
            print("No configs found.")
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Error',
                    'message': 'Set model configurations before running.',
                    'sticky': False,
                }
            }

        print('Model is running.')
        # Load your data into a pandas dataframe
        df = pd.read_csv("custom/supplysync/models/train.csv")

        # Convert the 'week' column to datetime format
        df['week'] = pd.to_datetime(df['week'])

        # Group data monthly and calculate monthly quantity sold per sku (sum the units sold)
        monthly_data = df.groupby(['sku_id', df['week'].dt.to_period('M')])['units_sold'].sum().reset_index()

        # Convert 'week' back to datetime for proper handling
        monthly_data['week'] = monthly_data['week'].dt.to_timestamp()

        # Separate data into training data (2011 to 2013)
        train_data = monthly_data[monthly_data['week'].between(start_ye, end_ye)]
        # old:    train_data = monthly_data[monthly_data['week'] <= '2013-0-01']

        # Dictionary to store evaluation metrics for each SKU
        evaluation_metrics = {}

        # List to store the results for CSV generation
        results_for_csv = []

        # Iterate over unique sku_ids
        for sku_id in monthly_data['sku_id'].unique():
            # Filter train data for the current sku_id
            train_sku_data = train_data[train_data['sku_id'] == sku_id]

            # Get additional columns for the current sku_id
            sku_info = df[df['sku_id'] == sku_id].iloc[0]
            sku_type = sku_info['type']
            category_l1 = sku_info['category_L1']
            category_l2 = sku_info['category_L2']
            vendor = sku_info['vendor']

            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_normalized = scaler.fit_transform(train_sku_data['units_sold'].values.reshape(-1, 1))

            # Prepare data for LSTM
            def create_sequences(data, sequence_length):
                x, y = [], []
                for i in range(len(data) - sequence_length):
                    x.append(data[i:(i + sequence_length)])
                    y.append(data[i + sequence_length])
                return np.array(x), np.array(y)

            sequence_length = 12
            x_train, y_train = create_sequences(train_normalized, sequence_length)

            root_path = os.path.join('custom', 'supplysync', 'models')
            file_path = os.path.join(root_path, 'lstm.pickle')

            # Check if the file exists
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    model = pickle.load(file)
            else:
                print("LSTM pickle not found.")
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Error',
                        'message': 'No model has been trained.',
                        'sticky': False,
                    }
                }

            # Forecast for 3 months after the latest date in the training data
            forecast_start_date = train_sku_data['week'].max() + pd.DateOffset(months=1)
            forecast_end_date = forecast_start_date + pd.DateOffset(months=3)
            num_forecast_months = 3

            # Generate forecasted values
            batch = train_normalized[-sequence_length:].reshape((1, sequence_length, 1))
            sku_forecasts = np.array([])  # Initialize an empty numpy array for forecasted values
            for i in range(num_forecast_months):
                predicted_value = model.predict(batch, verbose=0)[0][0]
                sku_forecasts = np.concatenate(
                    (sku_forecasts, np.array([predicted_value])))  # Concatenate predicted value
                predicted_value = np.array([[predicted_value]])
                batch = np.append(batch[:, 1:, :], predicted_value[:, np.newaxis], axis=1)

            sku_forecasts = scaler.inverse_transform(sku_forecasts.reshape(-1, 1))

            # Convert forecasted quantities to integers
            sku_forecasts = sku_forecasts.astype(int)

            for i in range(num_forecast_months):
                forecast_month = forecast_start_date + pd.DateOffset(months=i)
                results_for_csv.append({
                    "SKU": sku_id,
                    "Type": sku_type,
                    "Category_L1": category_l1,
                    "Category_L2": category_l2,
                    "Vendor": vendor,
                    "Forecasted_Quantity": sku_forecasts[i][0],  # Use integer forecasted quantity
                    "Forecast_Period": forecast_month.strftime('%d/%m/%y')  # Date in dd/mm/yy format
                })

        # Create DataFrame for CSV results
        results_df = pd.DataFrame(results_for_csv)

        # Calculate total forecasted quantity and determine rank
        results_df['Total_Forecasted_Qty'] = results_df.groupby('SKU')['Forecasted_Quantity'].transform('sum')
        results_df['Rank'] = results_df['Total_Forecasted_Qty'].rank(method='dense', ascending=True).astype(int)

        # Sort the results by Rank
        results_df = results_df.sort_values('Rank')

        # Drop the 'Total_Forecasted_Qty' column as it's no longer needed
        results_df = results_df.drop(columns=['Total_Forecasted_Qty'])

        # Save the forecasted results to a CSV file
        results_df.to_csv("custom/supplysync/models/forecasted_results.csv", index=False)
        print("Results saved to forecasted_results.csv")

        # code to write data from csv to odoo model
        self.search([]).unlink()
        with open('custom/supplysync/models/forecasted_results.csv', 'r') as file:
            csv_data = csv.DictReader(file)
            for row in csv_data:
                # Convert date from 'DD/MM/YYYY' to 'YYYY-MM-DD'
                forecasted_period = datetime.strptime(row['Forecast_Period'], '%d/%m/%y').date().strftime('%Y-%m-%d')

                # Use the corrected date format when creating the record
                self.create({
                    'sku': row['SKU'],
                    'type': row['Type'],
                    'category_L1': row['Category_L1'],
                    'category_L2': row['Category_L2'],
                    'vendor': row['Vendor'],
                    'foreQuant': row['Forecasted_Quantity'],
                    'forePeriod': forecasted_period,
                    'rank': row['Rank']
                })

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Success',
                'message': 'Model ran successfully!',
                'sticky': False,  # If true, the notification will require user interaction to dismiss
            }
        }

    @api.model
    def import_data_from_csv(self):
        # clear existing records
        self.search([]).unlink()
        with open('custom/supplysync/models/forecasted_results.csv', 'r') as file:
            csv_data = csv.DictReader(file)
            for row in csv_data:
                # Check and convert empty date strings to None
                fore_period = row['Forecast_Period'] if row['Forecast_Period'] else False

                # Similarly handle other fields that might be empty and need special handling
                fore_quant = int(row['Forecasted_Quantity']) if row['Forecasted_Quantity'] else False
                rank = int(row['Rank']) if row['Rank'] else False

                self.create({
                    'sku': row['SKU'],
                    'type': row['Type'],
                    'category_L1': row['Category_L1'],
                    'category_L2': row['Category_L2'],
                    'vendor': row['Vendor'],
                    'foreQuant': fore_quant,
                    'forePeriod': fore_period,
                    'rank': rank
                })


class ForecastConfig(models.Model):
    _name = "supplysync.forecast"
    _description = "Forecast Configuration"

    configid = fields.Integer("Config ID", required=True)
    train_end_date = fields.Date("Training End Date", required=True)
    train_start_date = fields.Date("Training Start Date", required=True)

    _sql_constraints = [
        ('configID_unique', 'unique(configid)', 'The Configuration ID must be unique.'),
    ]

    @api.model
    def set_config(self, ids):
        print('Configurations loaded successfully!')
        record = self.browse(ids)
        if record:
            global start_ye, end_ye
            start_ye = str(record.train_start_date)
            end_ye = str(record.train_end_date)
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Success',
                    'message': 'Configurations loaded successfully!',
                    'sticky': False,  # If true, the notification will require user interaction to dismiss
                }
            }

    @api.model
    def create(self, vals):
        # First, call the super to create the record
        record = super(ForecastConfig, self).create(vals)
        global start_ye, end_ye
        start_ye = str(record.train_start_date)
        end_ye = str(record.train_end_date)
        print(start_ye, end_ye)
        return record

    @api.model
    def train_model(self, ids):
        # check if configs are set
        if start_ye is None or end_ye is None:
            print("No configs found.")
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Error',
                    'message': 'Training cannot start. Model configurations are not set.',
                    'sticky': False,
                }
            }
        print("Training has started...")
        # Load your data into a pandas dataframe
        df = pd.read_csv("custom/supplysync/models/train.csv")

        # Convert the 'week' column to datetime format
        df['week'] = pd.to_datetime(df['week'])

        # Group data monthly and calculate monthly quantity sold per sku (sum the units sold)
        monthly_data = df.groupby(['sku_id', df['week'].dt.to_period('M')])['units_sold'].sum().reset_index()

        # Convert 'week' back to datetime for proper handling
        monthly_data['week'] = monthly_data['week'].dt.to_timestamp()

        # Separate data into training data (2011 to 2013)
        train_data = monthly_data[monthly_data['week'].between(start_ye, end_ye)]
        # old:    train_data = monthly_data[monthly_data['week'] <= '2013-0-01']

        # Dictionary to store evaluation metrics for each SKU
        evaluation_metrics = {}

        # List to store the results for CSV generation
        results_for_csv = []

        # Iterate over unique sku_ids
        for sku_id in monthly_data['sku_id'].unique():
            # Filter train data for the current sku_id
            train_sku_data = train_data[train_data['sku_id'] == sku_id]

            # Get additional columns for the current sku_id
            sku_info = df[df['sku_id'] == sku_id].iloc[0]
            sku_type = sku_info['type']
            category_l1 = sku_info['category_L1']
            category_l2 = sku_info['category_L2']
            vendor = sku_info['vendor']

            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_normalized = scaler.fit_transform(train_sku_data['units_sold'].values.reshape(-1, 1))

            # Prepare data for LSTM
            def create_sequences(data, sequence_length):
                x, y = [], []
                for i in range(len(data) - sequence_length):
                    x.append(data[i:(i + sequence_length)])
                    y.append(data[i + sequence_length])
                return np.array(x), np.array(y)

            sequence_length = 12
            x_train, y_train = create_sequences(train_normalized, sequence_length)

            # Define LSTM model
            model = Sequential([
                LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
                Dropout(0.5),  # Added dropout layer
                LSTM(units=50, kernel_regularizer=L1(0.05)),
                Dense(units=1)
            ])

            # Compile model
            model.compile(optimizer=SGD(learning_rate=0.001), loss='mean_squared_error')

            # Fit model
            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

            root_path = os.path.join('custom', 'supplysync', 'models')
            with open(os.path.join(root_path, 'lstm.pickle'), 'wb+') as file:
                pickle.dump(model, file)

        print("Model has been trained.")
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Success',
                'message': 'Model has been trained successfully!',
                'sticky': False,  # If true, the notification will require user interaction to dismiss
            }
        }
