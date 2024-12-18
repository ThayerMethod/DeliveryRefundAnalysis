import pandas as pd
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

def load_delivery_data(file_path):
    # Define the column names based on the provided input
    columns = [
        "Customer placed order datetime",
        "Placed order with restaurant datetime",
        "Driver at restaurant datetime",
        "Delivered to consumer datetime",
        "Driver ID",
        "Restaurant ID",
        "Consumer ID",
        "Delivery Region",
        "Is ASAP",
        "Order total",
        "Amount of discount",
        "Amount of tip",
        "Refunded amount"
    ]

    # Load the data from the Excel file
    data = pd.read_excel(file_path, usecols=columns)

    return data

# Example usage
if __name__ == "__main__":
    file_path = "CSV_DATA_URL"
    delivery_data = load_delivery_data(file_path)
    print(delivery_data.head())  # Print the first few rows to verify the data is loaded correctly

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_filter_data(file_path):
    columns = [
        "Customer placed order datetime",
        "Placed order with restaurant datetime",
        "Driver at restaurant datetime",
        "Delivered to consumer datetime",
        "Driver ID",
        "Restaurant ID",
        "Consumer ID",
        "Delivery Region",
        "Is ASAP",
        "Order total",
        "Amount of discount",
        "Amount of tip",
        "Refunded amount"
    ]

    # Load data
    data = pd.read_excel(file_path, usecols=columns)

    # Filter for Palo Alto and non-zero refunds
    data = data[(data['Delivery Region'] == 'Palo Alto') & (data['Refunded amount'] > 0)]

    # Parse datetime fields
    datetime_cols = [
        "Customer placed order datetime",
        "Placed order with restaurant datetime",
        "Driver at restaurant datetime",
        "Delivered to consumer datetime"
    ]
    for col in datetime_cols:
        data[col] = pd.to_datetime(data[col], format="%d %H:%M:%S")

    # Calculate True Total
    data['True Total'] = data['Order total'] - data['Amount of discount'] + data['Amount of tip']

    return data

def plot_data(data):
    # Histogram of refunded amounts
    plt.figure(figsize=(10, 5))
    plt.hist(data['Refunded amount'], bins=30, color='skyblue', alpha=0.7)
    mean_val = data['Refunded amount'].mean()
    median_val = data['Refunded amount'].median()
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.title('Histogram of Refunded Amounts in Palo Alto')
    plt.xlabel('Refunded Amount ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Scatter plot of True Total vs. Refunded Amount
    plt.figure(figsize=(10, 5))
    is_full_refund = data['Refunded amount'] == data['True Total']
    plt.scatter(data['True Total'][is_full_refund], data['Refunded amount'][is_full_refund], c='red', alpha=0.5, label='Full Refund (red)')
    plt.scatter(data['True Total'][~is_full_refund], data['Refunded amount'][~is_full_refund], c='green', alpha=0.5, label='Partial Refund (green)')
    plt.plot([data['True Total'].min(), data['True Total'].max()], [data['True Total'].min(), data['True Total'].max()], 'b--', lw=2)
    plt.title('True Total vs. Refunded Amount in Palo Alto')
    plt.xlabel('True Total ($)')
    plt.ylabel('Refunded Amount ($)')
    plt.legend(title="Refund Type")
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "CSV_DATA_URL"
    delivery_data = load_and_filter_data(file_path)
    plot_data(delivery_data)

import pandas as pd

def load_and_filter_data(file_path):
    columns = [
        "Customer placed order datetime",
        "Placed order with restaurant datetime",
        "Driver at restaurant datetime",
        "Delivered to consumer datetime",
        "Driver ID",
        "Restaurant ID",
        "Consumer ID",
        "Delivery Region",
        "Is ASAP",
        "Order total",
        "Amount of discount",
        "Amount of tip",
        "Refunded amount"
    ]

    # Load data
    data = pd.read_excel(file_path, usecols=columns)

    # Filter for Palo Alto and non-zero refunds
    data = data[(data['Delivery Region'] == 'Palo Alto') & (data['Refunded amount'] > 0)]

    # Calculate True Total
    data['True Total'] = data['Order total'] - data['Amount of discount'] + data['Amount of tip']

    return data

def calculate_refund_percentages(data):
    # Determine refund type
    data['Refund Type'] = 'Partial Refund'
    data.loc[data['Refunded amount'] == data['True Total'], 'Refund Type'] = 'Full Refund'

    # Calculate percentages
    total_refunds = len(data)
    full_refund_count = len(data[data['Refund Type'] == 'Full Refund'])
    partial_refund_count = len(data[data['Refund Type'] == 'Partial Refund'])

    full_refund_percentage = (full_refund_count / total_refunds) * 100
    partial_refund_percentage = (partial_refund_count / total_refunds) * 100

    return full_refund_percentage, partial_refund_percentage

# Example usage
if __name__ == "__main__":
    file_path = "CSV_DATA_URL"
    delivery_data = load_and_filter_data(file_path)
    full_refund_pct, partial_refund_pct = calculate_refund_percentages(delivery_data)
    print(f"Percentage of Full Refunds: {full_refund_pct:.2f}%")
    print(f"Percentage of Partial Refunds: {partial_refund_pct:.2f}%")

import pandas as pd

def load_and_filter_data(file_path):
    columns = [
        "Customer placed order datetime",
        "Placed order with restaurant datetime",
        "Driver at restaurant datetime",
        "Delivered to consumer datetime",
        "Driver ID",
        "Restaurant ID",
        "Consumer ID",
        "Delivery Region",
        "Is ASAP",
        "Order total",
        "Amount of discount",
        "Amount of tip",
        "Refunded amount"
    ]

    # Load data
    data = pd.read_excel(file_path, usecols=columns)

    # Filter for Palo Alto and non-zero refunds
    data = data[(data['Delivery Region'] == 'Palo Alto') & (data['Refunded amount'] > 0)]

    # Calculate True Total
    data['True Total'] = data['Order total'] - data['Amount of discount'] + data['Amount of tip']

    return data

def calculate_average_refund_percentage(data):
    # Compute refund percentages
    data['Refund Percentage'] = (data['Refunded amount'] / data['True Total']) * 100

    # Calculate the average of refund percentages
    average_refund_percentage = data['Refund Percentage'].mean()

    return average_refund_percentage

# Example usage
if __name__ == "__main__":
    file_path = "CSV_DATA_URL"
    delivery_data = load_and_filter_data(file_path)
    avg_refund_pct = calculate_average_refund_percentage(delivery_data)
    print(f"Average Percentage Refunded: {avg_refund_pct:.2f}%")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_filter_data(file_path):
    columns = [
        "Customer placed order datetime",
        "Delivered to consumer datetime",
        "Delivery Region",
        "Refunded amount"
    ]

    # Load data
    data = pd.read_excel(file_path, usecols=columns)

    # Filter for refunded orders in Palo Alto
    data = data[(data['Delivery Region'] == 'Palo Alto') & (data['Refunded amount'] > 0)]

    # Parse datetime fields
    data['Customer placed order datetime'] = pd.to_datetime(data['Customer placed order datetime'], format='%d %H:%M:%S')
    data['Delivered to consumer datetime'] = pd.to_datetime(data['Delivered to consumer datetime'], format='%d %H:%M:%S')

    # Calculate Time for Delivery in minutes
    data['Time for Delivery'] = (data['Delivered to consumer datetime'] - data['Customer placed order datetime']).dt.total_seconds() / 60

    # Remove entries with negative Time for Delivery and cap at 180 minutes
    data = data[data['Time for Delivery'] >= 0]
    data['Capped Time for Delivery'] = data['Time for Delivery'].clip(upper=180)

    return data

def plot_delivery_time_histogram(data):
    # Define bins with the last bin being for 180+ minutes
    bins = list(range(0, 181, 5)) + [np.inf]
    labels = [f"{i}-{i+4}" for i in range(0, 180, 5)] + ["180+"]

    # Cut the data into bins
    data['Time Bin'] = pd.cut(data['Capped Time for Delivery'], bins=bins, labels=labels, right=False)

    # Calculate mean, specific mean excluding 180+ minute deliveries, and median
    general_mean_time = data['Capped Time for Delivery'].mean()
    specific_mean_time = data[data['Time for Delivery'] < 180]['Time for Delivery'].mean()
    median_time = data['Capped Time for Delivery'].median()

    # Plot histogram
    plt.figure(figsize=(12, 6))
    counts = data['Time Bin'].value_counts(sort=False)
    counts.plot(kind='bar', color='skyblue', alpha=0.7)

    # Plot overall mean line
    overall_mean_bin_index = min(int(general_mean_time // 5), 36)
    plt.axvline(x=overall_mean_bin_index, color='blue', linestyle='dashed', linewidth=2, label=f'Overall Mean: {general_mean_time:.2f}')

    # Plot specific mean line excluding the 180+ capped values
    specific_mean_bin_index = min(int(specific_mean_time // 5), 36)
    plt.axvline(x=specific_mean_bin_index, color='red', linestyle='dashed', linewidth=2, label=f'Mean (Excl. 180+): {specific_mean_time:.2f}')

    # Plot median line
    median_bin_index = min(int(median_time // 5), 36)
    plt.axvline(x=median_bin_index, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_time:.2f}')

    plt.title('Histogram of Time for Delivery for Refunded Orders in Palo Alto')
    plt.xlabel('Time for Delivery (minutes)')
    plt.ylabel('Frequency of Refunds')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "CSV_DATA_URL"
    delivery_data = load_and_filter_data(file_path)
    plot_delivery_time_histogram(delivery_data)

import pandas as pd

def load_data(file_path):
    # Load the necessary columns
    columns = [
        "Customer placed order datetime",
        "Delivered to consumer datetime",
        "Is ASAP"  # This column identifies whether the delivery was ASAP or scheduled
    ]

    data = pd.read_excel(file_path, usecols=columns)
    # Convert datetime fields from string if necessary
    data['Customer placed order datetime'] = pd.to_datetime(data['Customer placed order datetime'], format='%d %H:%M:%S')
    data['Delivered to consumer datetime'] = pd.to_datetime(data['Delivered to consumer datetime'], format='%d %H:%M:%S')

    # Calculate Time for Delivery in minutes
    data['Time for Delivery'] = (data['Delivered to consumer datetime'] - data['Customer placed order datetime']).dt.total_seconds() / 60

    # Remove any negative values in 'Time for Delivery'
    data = data[data['Time for Delivery'] >= 0]

    return data

def check_scheduled_deliveries(data):
    # Filter for scheduled deliveries (assuming 'Is ASAP' being False indicates a scheduled delivery)
    scheduled_deliveries = data[data['Is ASAP'] == False]

    # Check if any scheduled deliveries took less than 180 minutes
    less_than_180 = scheduled_deliveries[scheduled_deliveries['Time for Delivery'] < 180]

    return not less_than_180.empty  # Returns True if there are any, False otherwise

# Example usage
if __name__ == "__main__":
    file_path = "CSV_DATA_URL"
    delivery_data = load_data(file_path)
    result = check_scheduled_deliveries(delivery_data)
    print(f"Are there any scheduled deliveries that took less than 180 minutes? {'Yes' if result else 'No'}")
