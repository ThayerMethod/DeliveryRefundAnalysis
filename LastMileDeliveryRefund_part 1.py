#Import
from google.colab import files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Upload a file
uploaded = files.upload()

# Load the CSV into a pandas DataFrame
df = pd.read_csv(list(uploaded.keys())[0])

# Display the DataFrame
df.head()

# Install necessary libraries (run this cell first in Colab)
!pip install xgboost imbalanced-learn

# Import necessary libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Data_CSV_URL')

# Create the target variable 'Refunded' (1 if Refund Amount > 0, else 0)
df['Refunded'] = df['Refund Amount'].apply(lambda x: 1 if x > 0 else 0)

# Drop the 'Refund Amount' column as it's no longer needed
df = df.drop('Refund Amount', axis=1)

# Handle missing values if any
# Fill missing numerical values with the mean
numerical_cols = ['Order Total', 'Amount of discount', 'Tip Total']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill missing categorical values with the mode
categorical_cols = ['Driver ID', 'Restaurant ID', 'Consumer ID', 'Delivery Region', 'Is ASAP']
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Convert 'Is ASAP' from boolean to integer (True=1, False=0)
df['Is ASAP'] = df['Is ASAP'].astype(int)

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
for col in ['Driver ID', 'Restaurant ID', 'Consumer ID', 'Delivery Region']:
    df[col] = le.fit_transform(df[col])

# Check class imbalance
print("Class distribution:")
print(df['Refunded'].value_counts())

# Define features (X) and target variable (y)
X = df.drop('Refunded', axis=1)
y = df['Refunded']

# Optionally sample a subset of the data for quicker processing
sample_fraction = 0.5  # Adjust this fraction as needed (e.g., 0.5 for 50% of data)
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=sample_fraction, stratify=y, random_state=42)

# Handle class imbalance using RandomUnderSampler (simpler and faster than SMOTE)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_sampled, y_sampled)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# Initialize cross-validation with fewer folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ==========================
# Hyperparameter Tuning for Random Forest
# ==========================
# Simplified parameter grid
param_dist_rf = {
    'n_estimators': [50],
    'max_depth': [None, 10],
    'min_samples_split': [5],
    'min_samples_leaf': [1],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist_rf,
    n_iter=5,  # Reduced number of iterations
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Train the model using RandomizedSearchCV
random_search_rf.fit(X_train, y_train)

# Best estimator
best_rf = random_search_rf.best_estimator_

print("\nBest Hyperparameters for Random Forest:")
print(random_search_rf.best_params_)

# Predict refunds on the test data
y_pred_rf = best_rf.predict(X_test)

# Evaluate the Random Forest model's performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
print(f"\nRandom Forest Model Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest ROC AUC: {roc_auc_rf:.4f}")

print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ==========================
# Hyperparameter Tuning for XGBoost
# ==========================
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)

# Simplified parameter grid
param_dist_xgb = {
    'n_estimators': [50],
    'max_depth': [3, 6],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}

random_search_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist_xgb,
    n_iter=5,  # Reduced number of iterations
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_xgb.fit(X_train, y_train)

# Best estimator
best_xgb = random_search_xgb.best_estimator_

print("\nBest Hyperparameters for XGBoost:")
print(random_search_xgb.best_params_)

# Predict refunds on the test data
y_pred_xgb = best_xgb.predict(X_test)

# Evaluate the XGBoost model's performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])
print(f"\nXGBoost Model Accuracy: {accuracy_xgb:.4f}")
print(f"XGBoost ROC AUC: {roc_auc_xgb:.4f}")

print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Choose the best model based on ROC AUC
if roc_auc_xgb > roc_auc_rf:
    print("\nXGBoost model performs better.")
    final_model = best_xgb
    y_pred = y_pred_xgb
    y_scores = best_xgb.predict_proba(X_test)[:, 1]
else:
    print("\nRandom Forest model performs better.")
    final_model = best_rf
    y_pred = y_pred_rf
    y_scores = best_rf.predict_proba(X_test)[:, 1]

# Get feature importances
importances = final_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the dataframe by importance descending
importance_df = importance_df.sort_values('Importance', ascending=False)

# Print the feature importances
print("\nFeature Importances:")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# ==========================
# Plotting ROC Curve and AUC
# ==========================

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ==========================
# Plotting Lift Chart
# ==========================

def plot_lift_curve(y_true, y_scores, num_bins=20):
    # Create a DataFrame with the true labels and predicted scores
    data = pd.DataFrame({'y_true': y_true, 'y_scores': y_scores})
    # Sort the data by predicted scores descending
    data = data.sort_values('y_scores', ascending=False).reset_index(drop=True)
    # Compute cumulative gains
    data['cumulative_events'] = data['y_true'].cumsum()
    data['cumulative_total'] = np.arange(1, len(data) + 1)
    data['cumulative_event_rate'] = data['cumulative_events'] / data['cumulative_total']
    # Compute baseline event rate
    baseline_rate = data['y_true'].mean()
    # Compute cumulative lift
    data['cumulative_lift'] = data['cumulative_event_rate'] / baseline_rate
    # Compute cumulative percentile
    data['cumulative_percentile'] = (data['cumulative_total'] / data['cumulative_total'].max()) * 100
    # Downsample for plotting
    plot_data = data.iloc[::max(1, len(data) // num_bins), :]
    # Plot the cumulative lift curve
    plt.figure(figsize=(8, 6))
    plt.plot(plot_data['cumulative_percentile'], plot_data['cumulative_lift'], marker='o', linestyle='-')
    plt.xlabel('Cumulative Percentile')
    plt.ylabel('Cumulative Lift')
    plt.title('Cumulative Lift Chart')
    plt.grid(True)
    plt.show()

# Call the function to plot the lift curve
plot_lift_curve(y_test.reset_index(drop=True), y_scores)

# Install necessary libraries
!pip install networkx

# Import necessary libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Data_CSV_URL')

# Filter for Palo Alto data
df_pa = df[df['Delivery Region'] == 'Palo Alto']

# Create 'Refunded' column (1 if Refund Amount > 0, else 0)
df_pa['Refunded'] = df_pa['Refund Amount'].apply(lambda x: 1 if x > 0 else 0)

# Filter for refunded transactions
df_refunded = df_pa[df_pa['Refunded'] == 1]

# Check if there are any refunded transactions after filtering
if df_refunded.empty:
    print("No refunded transactions found in Palo Alto.")
else:
    # Build the graph
    G = nx.MultiGraph()  # Use MultiGraph to allow multiple edges between nodes

    # Prefix IDs to avoid conflicts
    def prefix_id(id_value, prefix):
        return f"{prefix}_{id_value}"

    # Node counts
    restaurant_counts = df_refunded['Restaurant ID'].value_counts()
    driver_counts = df_refunded['Driver ID'].value_counts()
    customer_counts = df_refunded['Consumer ID'].value_counts()

    # Add Restaurants to the graph
    for restaurant_id, count in restaurant_counts.items():
        node_id = prefix_id(restaurant_id, 'Restaurant')
        G.add_node(node_id, type='Restaurant', size=count)

    # Add Drivers to the graph
    for driver_id, count in driver_counts.items():
        node_id = prefix_id(driver_id, 'Driver')
        G.add_node(node_id, type='Driver', size=count)

    # Add Customers to the graph
    for customer_id, count in customer_counts.items():
        node_id = prefix_id(customer_id, 'Customer')
        G.add_node(node_id, type='Customer', size=count)

    # Edge counts between Restaurant and Driver
    restaurant_driver_edges = df_refunded.groupby(['Restaurant ID', 'Driver ID']).size().reset_index(name='count')

    # Add edges between Restaurants and Drivers
    for idx, row in restaurant_driver_edges.iterrows():
        restaurant_id = prefix_id(row['Restaurant ID'], 'Restaurant')
        driver_id = prefix_id(row['Driver ID'], 'Driver')
        count = row['count']
        G.add_edge(restaurant_id, driver_id, color='brown', weight=count, curvature=0.2)

    # Edge counts between Driver and Customer
    driver_customer_edges = df_refunded.groupby(['Driver ID', 'Consumer ID']).size().reset_index(name='count')

    # Add edges between Drivers and Customers
    for idx, row in driver_customer_edges.iterrows():
        driver_id = prefix_id(row['Driver ID'], 'Driver')
        customer_id = prefix_id(row['Consumer ID'], 'Customer')
        count = row['count']
        G.add_edge(driver_id, customer_id, color='black', weight=count, curvature=0.2)

    # Prepare for drawing
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)

    # Get lists of nodes by type
    restaurants = [n for n, attr in G.nodes(data=True) if attr['type'] == 'Restaurant']
    drivers = [n for n, attr in G.nodes(data=True) if attr['type'] == 'Driver']
    customers = [n for n, attr in G.nodes(data=True) if attr['type'] == 'Customer']

    # Adjust node sizes relative to their own category
    # Get maximum counts for normalization
    max_restaurant_count = max([G.nodes[n]['size'] for n in restaurants]) if restaurants else 1
    max_driver_count = max([G.nodes[n]['size'] for n in drivers]) if drivers else 1
    max_customer_count = max([G.nodes[n]['size'] for n in customers]) if customers else 1

    # Define scaling factors for each category
    restaurant_scaling_factor = 800
    driver_scaling_factor = 800
    customer_scaling_factor = 800

    # Compute node sizes
    restaurant_sizes = [(G.nodes[n]['size'] / max_restaurant_count) * restaurant_scaling_factor for n in restaurants]
    driver_sizes = [(G.nodes[n]['size'] / max_driver_count) * driver_scaling_factor for n in drivers]
    customer_sizes = [(G.nodes[n]['size'] / max_customer_count) * customer_scaling_factor for n in customers]

    # Get edges by color
    brown_edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'brown']
    brown_weights = [G[u][v][0]['weight'] for u, v in brown_edges]

    black_edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'black']
    black_weights = [G[u][v][0]['weight'] for u, v in black_edges]

    # Adjust edge widths (increased scaling factor)
    edge_scaling_factor = 2  # Adjust this value to make edges more prominent
    brown_weights = [w * edge_scaling_factor for w in brown_weights]
    black_weights = [w * edge_scaling_factor for w in black_weights]

    # Plot the graph
    plt.figure(figsize=(20, 20))

    # Draw edges with curvature
    arc_rad = 0.2  # Adjust curvature radius
    nx.draw_networkx_edges(G, pos, edgelist=brown_edges, edge_color='brown', width=brown_weights,
                           connectionstyle=f'arc3, rad = {arc_rad}')
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color='black', width=black_weights,
                           connectionstyle=f'arc3, rad = -{arc_rad}')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=restaurants, node_color='red', node_shape='s', node_size=restaurant_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=drivers, node_color='orange', node_shape='*', node_size=driver_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='green', node_shape='o', node_size=customer_sizes)

    # Optionally, add labels (commented out to avoid clutter)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    # Remove axes
    plt.axis('off')

    # Show the plot without title and legend
    plt.show()

# Import necessary libraries
import pandas as pd

# Load the data
df = pd.read_csv('Data_CSV_URL')

# Create 'Refunded' column (1 if Refund Amount > 0, else 0)
df['Refunded'] = df['Refund Amount'].apply(lambda x: 1 if x > 0 else 0)

# Filter for Palo Alto data (if needed)
# Uncomment the next line to filter for Palo Alto
# df = df[df['Delivery Region'] == 'Palo Alto']

# Create a dataframe with only refunded transactions
df_refunded = df[df['Refunded'] == 1]

# ============================================
# Calculate statistics for Restaurants
# ============================================

# Aggregate data for restaurants
restaurant_stats = df.groupby('Restaurant ID').agg(
    total_transactions=('Refunded', 'count'),           # Total number of transactions per restaurant
    total_refunded_transactions=('Refunded', 'sum')     # Total number of refunded transactions per restaurant
).reset_index()

# Calculate refund rate
restaurant_stats['refund_rate'] = (restaurant_stats['total_refunded_transactions'] / restaurant_stats['total_transactions']) * 100

# Calculate the number of unique customers who requested refunds per restaurant
unique_customers_refunded = df_refunded.groupby('Restaurant ID')['Consumer ID'].nunique().reset_index()
unique_customers_refunded.rename(columns={'Consumer ID': 'unique_customers_refunded'}, inplace=True)

# Merge the unique customer counts into restaurant_stats
restaurant_stats = pd.merge(restaurant_stats, unique_customers_refunded, on='Restaurant ID', how='left')

# Fill NaN values with 0 (for restaurants with no refunds)
restaurant_stats['unique_customers_refunded'] = restaurant_stats['unique_customers_refunded'].fillna(0).astype(int)

# Get top 5 Restaurants by number of refunded transactions
top5_restaurants = restaurant_stats.sort_values('total_refunded_transactions', ascending=False).head(5)

# ============================================
# Calculate statistics for Drivers
# ============================================

# Aggregate data for drivers
driver_stats = df.groupby('Driver ID').agg(
    total_transactions=('Refunded', 'count'),           # Total number of transactions per driver
    total_refunded_transactions=('Refunded', 'sum')     # Total number of refunded transactions per driver
).reset_index()

# Calculate refund rate
driver_stats['refund_rate'] = (driver_stats['total_refunded_transactions'] / driver_stats['total_transactions']) * 100

# Get top 5 Drivers by number of refunded transactions
top5_drivers = driver_stats.sort_values('total_refunded_transactions', ascending=False).head(5)

# ============================================
# Calculate statistics for Customers
# ============================================

# Aggregate data for customers
customer_stats = df.groupby('Consumer ID').agg(
    total_transactions=('Refunded', 'count'),           # Total number of transactions per customer
    total_refunded_transactions=('Refunded', 'sum')     # Total number of refunded transactions per customer
).reset_index()

# Calculate refund rate
customer_stats['refund_rate'] = (customer_stats['total_refunded_transactions'] / customer_stats['total_transactions']) * 100

# Calculate the number of unique restaurants from which each customer requested refunds
unique_restaurants_refunded = df_refunded.groupby('Consumer ID')['Restaurant ID'].nunique().reset_index()
unique_restaurants_refunded.rename(columns={'Restaurant ID': 'unique_restaurants_refunded_from'}, inplace=True)

# Merge the unique restaurant counts into customer_stats
customer_stats = pd.merge(customer_stats, unique_restaurants_refunded, on='Consumer ID', how='left')

# Fill NaN values with 0 (for customers with no refunds)
customer_stats['unique_restaurants_refunded_from'] = customer_stats['unique_restaurants_refunded_from'].fillna(0).astype(int)

# Get top 5 Customers by number of refunded transactions
top5_customers = customer_stats.sort_values('total_refunded_transactions', ascending=False).head(5)

# ============================================
# Display the results
# ============================================

print("Top 5 Restaurants with the Most Refunded Transactions:")
print(top5_restaurants[['Restaurant ID', 'total_transactions', 'total_refunded_transactions', 'refund_rate', 'unique_customers_refunded']])

print("\nTop 5 Drivers with the Most Refunded Transactions:")
print(top5_drivers[['Driver ID', 'total_transactions', 'total_refunded_transactions', 'refund_rate']])

print("\nTop 5 Customers with the Most Refunded Transactions:")
print(top5_customers[['Consumer ID', 'total_transactions', 'total_refunded_transactions', 'refund_rate', 'unique_restaurants_refunded_from']])

# Import necessary libraries
import pandas as pd

# Load the data
df = pd.read_csv('Data_CSV_URL')

# Filter for Palo Alto data
df_palo_alto = df[df['Delivery Region'] == 'Palo Alto']

# Count unique Customers
num_customers = df_palo_alto['Consumer ID'].nunique()

# Count unique Restaurants
num_restaurants = df_palo_alto['Restaurant ID'].nunique()

# Count unique Drivers
num_drivers = df_palo_alto['Driver ID'].nunique()

# Display the results
print(f"Number of unique customers in Palo Alto: {num_customers}")
print(f"Number of unique restaurants in Palo Alto: {num_restaurants}")
print(f"Number of unique drivers in Palo Alto: {num_drivers}")

# Install necessary libraries
!pip install networkx

# Import necessary libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Data_CSV_URL')

# Filter for Palo Alto data (if needed)
df_pa = df[df['Delivery Region'] == 'Palo Alto']

# Create 'Refunded' column (1 if Refund Amount > 0, else 0)
df_pa['Refunded'] = df_pa['Refund Amount'].apply(lambda x: 1 if x > 0 else 0)

# Filter for refunded transactions
df_refunded = df_pa[df_pa['Refunded'] == 1]

# Check if there are any refunded transactions after filtering
if df_refunded.empty:
    print("No refunded transactions found in Palo Alto.")
else:
    # Build the graph
    G = nx.Graph()

    # Prefix IDs to avoid conflicts (if necessary)
    def prefix_id(id_value, prefix):
        return f"{prefix}_{id_value}"

    # Node counts
    restaurant_counts = df_refunded['Restaurant ID'].value_counts()
    customer_counts = df_refunded['Consumer ID'].value_counts()

    # Add Restaurants to the graph
    for restaurant_id, count in restaurant_counts.items():
        node_id = prefix_id(restaurant_id, 'Restaurant')
        G.add_node(node_id, type='Restaurant', size=count)

    # Add Customers to the graph
    for customer_id, count in customer_counts.items():
        node_id = prefix_id(customer_id, 'Customer')
        G.add_node(node_id, type='Customer', size=count)

    # Edge counts between Restaurant and Customer
    restaurant_customer_edges = df_refunded.groupby(['Restaurant ID', 'Consumer ID']).size().reset_index(name='count')

    # Add edges between Restaurants and Customers
    for idx, row in restaurant_customer_edges.iterrows():
        restaurant_id = prefix_id(row['Restaurant ID'], 'Restaurant')
        customer_id = prefix_id(row['Consumer ID'], 'Customer')
        count = row['count']
        G.add_edge(restaurant_id, customer_id, weight=count)

    # Prepare for drawing

    # Option 1: Use Kamada-Kawai layout for better node separation
    pos = nx.kamada_kawai_layout(G)

    # Option 2: Adjust the spring_layout parameters for more separation
    # pos = nx.spring_layout(G, k=1.0, iterations=200, seed=42)

    # Get lists of nodes by type
    restaurants = [n for n, attr in G.nodes(data=True) if attr['type'] == 'Restaurant']
    customers = [n for n, attr in G.nodes(data=True) if attr['type'] == 'Customer']

    # Adjust node sizes relative to their own category
    # Get maximum counts for normalization
    max_restaurant_count = max([G.nodes[n]['size'] for n in restaurants]) if restaurants else 1
    max_customer_count = max([G.nodes[n]['size'] for n in customers]) if customers else 1

    # Define scaling factors for each category
    restaurant_scaling_factor = 800
    customer_scaling_factor = 800

    # Compute node sizes
    restaurant_sizes = [(G.nodes[n]['size'] / max_restaurant_count) * restaurant_scaling_factor for n in restaurants]
    customer_sizes = [(G.nodes[n]['size'] / max_customer_count) * customer_scaling_factor for n in customers]

    # Get edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Adjust edge widths (increase scaling factor if needed)
    edge_scaling_factor = 2
    edge_widths = [w * edge_scaling_factor for w in edge_weights]

    # Plot the graph
    plt.figure(figsize=(25, 25))

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='brown', alpha=0.5)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=restaurants, node_color='red', node_shape='s', node_size=restaurant_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='green', node_shape='o', node_size=customer_sizes)

    # Optionally, add labels (commented out to avoid clutter)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    # Remove axes
    plt.axis('off')

    # Show the plot without title and legend
    plt.show()

# Install necessary libraries
!pip install pandas matplotlib numpy

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('Data_CSV_URL')

# Create 'Refunded' column (1 if Refund Amount > 0, else 0)
df['Refunded'] = df['Refund Amount'].apply(lambda x: 1 if x > 0 else 0)

# ============================================
# Step 1: Calculate Customer Statistics
# ============================================

# Total number of orders and total refund amount per customer
customer_stats = df.groupby('Consumer ID').agg(
    total_orders=('Consumer ID', 'count'),
    total_refund_amount=('Refund Amount', 'sum')
).reset_index()

# Sort customers by total_orders descending
customer_stats = customer_stats.sort_values(by='total_orders', ascending=False).reset_index(drop=True)

# Assign quintiles using numpy.array_split to ensure exactly 5 bins
quintiles = np.array_split(customer_stats, 5)

# Initialize 'transaction_quintile' column
customer_stats['transaction_quintile'] = 0

# Assign quintile labels (0 to 4)
for i, quintile in enumerate(quintiles):
    customer_stats.loc[quintile.index, 'transaction_quintile'] = i

# Merge quintile labels back to the main dataframe
df = df.merge(customer_stats[['Consumer ID', 'transaction_quintile']], on='Consumer ID', how='left')

# Drop rows with NaN in 'transaction_quintile' (if any)
df = df.dropna(subset=['transaction_quintile'])

# Ensure 'transaction_quintile' is integer type
df['transaction_quintile'] = df['transaction_quintile'].astype(int)

# ============================================
# Step 2: Handle 'Is ASAP' Missing Values
# ============================================

# Fill missing 'Is ASAP' values with 'Missing'
df['Is ASAP'] = df['Is ASAP'].fillna('Missing')

# ============================================
# Step 3: Calculate Refund Amounts by Quintile and 'Is ASAP'
# ============================================

# Total refund amount per quintile and 'Is ASAP' status
refunds_quintile_asap = df.groupby(['transaction_quintile', 'Is ASAP']).agg(
    total_refund_amount=('Refund Amount', 'sum')
).reset_index()

# Total refund amount per quintile
refunds_per_quintile = df.groupby('transaction_quintile').agg(
    quintile_refund_amount=('Refund Amount', 'sum')
).reset_index()

# Merge total refund amount per quintile into refunds_quintile_asap
refunds_quintile_asap = refunds_quintile_asap.merge(
    refunds_per_quintile, on='transaction_quintile', how='left'
)

# Calculate percentage of refund amount per 'Is ASAP' status within each quintile
refunds_quintile_asap['refund_percentage'] = (
    refunds_quintile_asap['total_refund_amount'] / refunds_quintile_asap['quintile_refund_amount']
) * 100

# ============================================
# Step 4: Calculate Number of Customers per Quintile
# ============================================

# Number of customers per quintile
customers_per_quintile = customer_stats.groupby('transaction_quintile').agg(
    num_customers=('Consumer ID', 'count')
).reset_index()

# Total number of customers
total_customers = customer_stats['Consumer ID'].nunique()

# Calculate width percentage for each quintile
customers_per_quintile['width_percentage'] = (customers_per_quintile['num_customers'] / total_customers) * 100

# ============================================
# Step 5: Prepare Data for Plotting
# ============================================

# Merge refunds and widths
plot_data = refunds_quintile_asap.merge(
    customers_per_quintile[['transaction_quintile', 'width_percentage']],
    on='transaction_quintile',
    how='left'
)

# Sort data
plot_data.sort_values(['transaction_quintile', 'Is ASAP'], inplace=True)

# ============================================
# Step 6: Define Colors for 'Is ASAP'
# ============================================

# Get unique 'Is ASAP' values
asap_values = plot_data['Is ASAP'].unique()

# Define colors for 'Is ASAP', making 'Missing' grey
colors = plt.cm.Set1.colors  # Use a colormap with distinct colors
asap_colors = {}
for i, value in enumerate(asap_values):
    if value == 'Missing':
        asap_colors[value] = 'grey'
    else:
        asap_colors[value] = colors[i % len(colors)]

# ============================================
# Step 7: Create the Mekko Chart
# ============================================

# Initialize the plot
plt.figure(figsize=(15, 8))

# Plot each quintile as a variable-width bar
left = 0  # Starting position on the x-axis

for bin_label in sorted(plot_data['transaction_quintile'].unique()):
    bin_data = plot_data[plot_data['transaction_quintile'] == bin_label]

    # Total width for this quintile
    width = customers_per_quintile.loc[customers_per_quintile['transaction_quintile'] == bin_label, 'width_percentage'].values[0]

    # Heights and 'Is ASAP' statuses
    heights = bin_data['refund_percentage'].values
    asap_statuses = bin_data['Is ASAP'].values

    # Cumulative height for stacking
    bottom = 0

    for i in range(len(heights)):
        height = heights[i]
        asap_status = asap_statuses[i]
        plt.bar(
            x=left,  # Use 'x' instead of 'left' to specify the x-coordinate
            height=height,
            width=width,
            bottom=bottom,
            color=asap_colors[asap_status],
            edgecolor='white'
        )
        bottom += height  # Update bottom for stacking

    # Update left for the next quintile
    left += width

# Set labels and title
plt.xlabel('Customer Quintiles (by Number of Orders Made)')
plt.ylabel('Percentage of Refund Amount per Quintile')
plt.title('Mekko Chart: Refund Amount by Customer Quintiles and Is ASAP')

# Set y-axis limits to 0 to 100%
plt.ylim(0, 100)

# Calculate midpoints for x-axis labels
bin_positions = customers_per_quintile['width_percentage'].cumsum() - (customers_per_quintile['width_percentage'] / 2)

plt.xticks(
    bin_positions,
    ['Quintile {}'.format(int(q) + 1) for q in customers_per_quintile['transaction_quintile']],
    rotation=0
)

# Create custom legend
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=asap_colors[value], label=f'Is ASAP: {value}') for value in asap_values]
plt.legend(handles=legend_elements, title='Is ASAP', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
