import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from odoo import api, fields, models


class Sku(models.Model):
    _name = "supplysync.sku"
    _description = "SupplySync SKU"

    # product input fields
    sku = fields.Char("Sku")
    type = fields.Char("Type")
    category_L1 = fields.Char("Category_L1")
    category_L2 = fields.Char("Category_L2")
    vendor = fields.Char("Vendor")
    rank = fields.Integer("Rank")

    @api.model
    def run_model(self):
        # Load your data into a pandas dataframe
        df = pd.read_csv("custom/supplysync/models/train.csv")

        # Convert the 'week' column to datetime format
        df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')

        # Group data monthly and calculate monthly quantity sold per sku (sum the units sold)
        monthly_data = df.groupby(['sku_id', df['week'].dt.to_period('M')])['units_sold'].sum()

        # Reset index to access sku_id and week as columns
        monthly_data = monthly_data.reset_index()

        # Separate data into train (2011 and 2012) and test (2013)
        train_data = monthly_data[monthly_data['week'] < '2013']
        test_data = monthly_data[monthly_data['week'] >= '2013']

        # Dictionary to store evaluation metrics for each SKU
        evaluation_metrics = {}

        # List to store the results for CSV generation
        results_for_csv = []

        # Function to create sequences for LSTM
        def create_sequences(data, sequence_length):
            x, y = [], []
            for i in range(len(data) - sequence_length):
                x.append(data[i:(i + sequence_length)])
                y.append(data[i + sequence_length])
            return np.array(x), np.array(y)

        # Iterate over unique sku_ids
        for sku_id in monthly_data['sku_id'].unique():
            # Filter train and test data for the current sku_id
            train_sku_data = train_data[train_data['sku_id'] == sku_id]
            test_sku_data = test_data[test_data['sku_id'] == sku_id]

            if len(test_sku_data) == 0 or len(train_sku_data) == 0:
                continue  # Skip if there's no test data or train data for this SKU

            # Normalize data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_normalized = scaler.fit_transform(train_sku_data['units_sold'].values.reshape(-1, 1))

            # Prepare data for LSTM
            sequence_length = 12
            x_train, y_train = create_sequences(train_normalized, sequence_length)

            # Define LSTM model
            model = Sequential([
                LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
                Dropout(0.2),  # Added dropout layer
                LSTM(units=50),
                Dense(units=1)
            ])

            # Compile model
            model.compile(optimizer=Adam(), loss='mean_squared_error')

            # Fit model
            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

            # Forecast using LSTM model
            forecasted_values_lstm = []
            batch = train_normalized[-sequence_length:].reshape((1, sequence_length, 1))
            for i in range(len(test_sku_data)):
                predicted_value = model.predict(batch)[0][0]
                forecasted_values_lstm.append(predicted_value)

                # Reshape predicted value to match the dimensions of the last slice of batch
                predicted_value = np.array([[predicted_value]])

                # Append predicted value to batch after reshaping
                batch = np.append(batch[:, 1:, :], predicted_value[:, np.newaxis], axis=1)

            # Denormalize forecasted values for LSTM
            forecasted_values_lstm = scaler.inverse_transform(np.array(forecasted_values_lstm).reshape(-1, 1))

            # ----------------------- SVM STARTS HERE ------------------------

            # Prepare data for SVM
            x_train_svm = train_sku_data['units_sold'].values[:-1].reshape(-1, 1)
            y_train_svm = train_sku_data['units_sold'].values[1:]
            x_test_svm = test_sku_data['units_sold'].values.reshape(-1, 1)  # Corrected this to use all test data points

            # Normalize data for SVM
            scaler_svm = StandardScaler()
            x_train_svm = scaler_svm.fit_transform(x_train_svm)
            x_test_svm = scaler_svm.transform(x_test_svm)

            # Train SVM model
            svm_model = SVR(kernel='rbf')
            svm_model.fit(x_train_svm, y_train_svm)

            # Forecast using SVM model
            forecasted_values_svm = svm_model.predict(x_test_svm).reshape(-1, 1)

            # Ensure both LSTM and SVM predictions have the same length
            forecasted_values_lstm = forecasted_values_lstm[:len(forecasted_values_svm)]
            forecasted_values_svm = forecasted_values_svm[:len(forecasted_values_lstm)]

            # Combine LSTM and SVM predictions using average
            forecasted_values = (forecasted_values_lstm + forecasted_values_svm) / 2

            # Calculate evaluation metrics
            true_values = test_sku_data['units_sold'].values[:len(forecasted_values)]
            mae = mean_absolute_error(true_values, forecasted_values)
            mse = mean_squared_error(true_values, forecasted_values)
            mape = mean_absolute_percentage_error(true_values, forecasted_values)
            r2 = r2_score(true_values, forecasted_values)

            # Store evaluation metrics for the current sku_id
            evaluation_metrics[sku_id] = {'MAE': mae, 'MSE': mse, 'MAPE': mape, 'R2': r2}

            # Collect data for CSV
            for i in range(len(test_sku_data)):
                results_for_csv.append({
                    "SKU": sku_id,
                    "Date": test_sku_data.iloc[i]['week'].strftime('%Y-%m'),
                    "Actual Value": true_values[i],
                    "Forecasted Value": forecasted_values[i][0]
                })

            # Print SKU, month, actual value, forecasted value, and evaluation metrics
            for i in range(len(test_sku_data)):
                print(f"SKU: {sku_id}, Month: {test_sku_data.iloc[i]['week'].strftime('%Y-%m')}, "
                      f"Actual Value: {true_values[i]}, Forecasted Value: {forecasted_values[i][0]:.2f}, "
                      f"MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape * 100:.2f}%, R2: {r2:.2f}")

        # Calculate overall average MAE, MSE, MAPE, and R2
        overall_avg_mae = np.mean([metrics['MAE'] for metrics in evaluation_metrics.values()])
        overall_avg_mse = np.mean([metrics['MSE'] for metrics in evaluation_metrics.values()])
        overall_avg_mape = np.mean([metrics['MAPE'] for metrics in evaluation_metrics.values()])
        overall_avg_r2 = np.mean([metrics['R2'] for metrics in evaluation_metrics.values()])
        print(f"Overall Average MAE: {overall_avg_mae:.2f}")
        print(f"Overall Average MSE: {overall_avg_mse:.2f}")
        print(f"Overall Average MAPE: {overall_avg_mape * 100:.2f}%")
        print(f"Overall Average R2: {overall_avg_r2:.2f}")

        # Generate CSV file with results
        results_df = pd.DataFrame(results_for_csv)
        results_df.to_csv("custom/supplysync/models/forecasted_results.csv", index=False)
        print("Results saved to custom/supplysync/models/forecasted_results.csv")

    @api.model
    def import_data_from_csv(self):
        # clear existing records
        self.search([]).unlink()
        with open('custom/supplysync/models/temp.csv', 'r') as file:
            csv_data = csv.DictReader(file)
            for row in csv_data:
                self.create({'sku': row['sku'],
                             'type': row['type'],
                             'category_L1': row['category_L1'],
                             'category_L2': row['category_L2'],
                             'vendor': row['vendor'],
                             'rank': int(row['rank'])
                             })

class ForecastConfig(models.Model):
    _name = "supplysync.forecast"
    _description = "Forecast Configuration"

    configID = fields.Integer()
    train_end_year = fields.Integer("Training End Year")
    test_start_year = fields.Integer("Testing Start Year")

    _sql_constraints = [
        ('configID_unique', 'UNIQUE(configID)', 'The Configuration ID must be unique.')
    ]

# class Forecast(models.Model):
#     _name = "supplysync.forecast"
#     _description = "SupplySync Forecast"
#
#     sku_id = fields.Many2one("supplysync.sku", "SKU")
#     forecasted_quantity = fields.Integer("Forecasted Quantity")
#     forecasted_date = fields.Date("Forecasted Date")
#
#     @api.depends(
#         "sku_id",
#         "forecasted_quantity",
#         "forecasted_date",
#     )
#     def save_forecast(self):
#         """
#         This method should save the forecasted data to the database.
#         """
#         for _ in self:
#             # Your saving logic here
#             pass
