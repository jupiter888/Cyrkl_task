import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences


# PREPROCESSING DATA
# load data
customers = pd.read_csv('/Users/_/Desktop/pythonProject/cyrkl_task_data(1)/customers.csv')
offers = pd.read_csv('/Users/_/Desktop/pythonProject/cyrkl_task_data(1)/offers.csv')
buyer_activity = pd.read_csv('/Users/_/Desktop/pythonProject/cyrkl_task_data(1)/buyer_activity.csv')
# identify missing values dataframes
print("Missing values in Customers:", customers.isnull().sum(), sep=',')
print("\nMissing values in Offers:", offers.isnull().sum(), sep=',')
print("\nMissing values in Buyer Activity:", buyer_activity.isnull().sum(), sep=',')
# time constants
DATE_START = datetime.strptime("2023-08-07", "%Y-%m-%d")
DATE_END = datetime.strptime("2023-08-13", "%Y-%m-%d")
# convert strings to datetime format
offers['created_at'] = pd.to_datetime(offers['created_at'])
buyer_activity['created_at'] = pd.to_datetime(buyer_activity['created_at'])
customers['created_at'] = pd.to_datetime(customers['created_at'])
# filter rows based on date range for previous week
offers = offers[(offers['created_at'] >= DATE_START) & (offers['created_at'] <= DATE_END)]
#buyer_activity = buyer_activity[(buyer_activity['created_at'] >= DATE_START) & (buyer_activity['created_at'] <= DATE_END)]  #empty df???????????
print(buyer_activity.columns, buyer_activity)
# cleaning data
# filtering out unwanted data from customers(unactivated & scammers)
customers = customers[(customers['is_blocked'] == False) & (customers['is_activated'] == True)]
# print the cleaned DataFrames to check
print(customers, customers.columns)
# check in customers df if there is any data in column deleted_at
non_empty_rows = customers[customers['deleted_at'].notna()]
if not non_empty_rows.empty:
    print("There are rows with data in the 'deleted_at' column.")
else:
    print("There are no rows with data in the 'deleted_at' column.")
# remove unused columns from customers (deleted_at, is_activated, is_blocked)
customers = customers.drop(columns=['deleted_at', 'is_activated', 'is_blocked'])
print(customers.columns)
# preprocessing data including cleaning: COMPLETED


# FEATURE ENGINEERING
# merge data
# the transformer model requires a sequence of events to predict user behavior. So, we need to:
# sort data: Order user activities by the created_at column to get a chronological sequence of actions.
# create user sequences: for each user, form a sequence of their activities. Each activity can be represented as a concatenation of relevant columns like item_type, product_form, category, sub_category, etc.
# encode categorical features: convert categorical variables like item_type, product_form, etc., into numerical form, probably using embeddings.
# sequence padding: ensure that all user sequences have the same length by padding shorter sequences.

# MERGING
# preparing to join the customers and offers together
print(offers.columns)
print(customers.columns)
print(buyer_activity.columns)
# check Unique IDs in Each DataFrame:
# account IDs and offer IDs exist in both df's?
print(customers['account_id'].nunique())
print(offers['account_id'].nunique())
print(buyer_activity['account_id'].nunique())#0
print(offers['offer_id'].nunique())
print(buyer_activity['offer_id'].nunique())#0
#overlapping_account_ids_in_customers_and_offers will hold a set of account_ids that are present in both the customers and offers df's
overlapping_account_ids_in_customers_and_offers = set(customers['account_id']).intersection(set(offers['account_id']))
overlapping_account_ids_in_offers_and_activity = set(offers['account_id']).intersection(set(buyer_activity['account_id']))
overlapping_offer_ids_in_offers_and_activity = set(offers['offer_id']).intersection(set(buyer_activity['offer_id']))
print("Number of overlapping account_ids between customers and offers:", len(overlapping_account_ids_in_customers_and_offers)) #18
print("Number of overlapping account_ids between offers and buyer_activity:", len(overlapping_account_ids_in_offers_and_activity)) #0
print("Number of overlapping offer_ids between offers and buyer_activity:", len(overlapping_offer_ids_in_offers_and_activity)) #0
# if the number of overlapping IDs is greater than 0, then there's overlap between the df's for those IDs.
# if 0 then no overlap, which would explain why the merged df has no rows when using an inner join.
# merge the customers & offers tables on account_id:
merged_data = pd.merge(customers, offers, on='account_id', how='inner')
print(merged_data, merged_data.columns)
# merge the above merged_data with buyer_activity on both account_id and offer_id:
final_merged_data = pd.merge(merged_data, buyer_activity, on=['account_id', 'offer_id'], how='inner')
# final_merged_data contains the merged data from all three tables.
print(final_merged_data.head())

# progress line: all below not finished
# Sort data
final_merged_data.sort_values(by='created_at')

# create user sequences
from sklearn.preprocessing import LabelEncoder
user_sequences = merged_data.groupby('account_id').apply(lambda x: x[['item_type', 'product_form', 'category', 'sub_category']].apply(lambda y: '_'.join(y.astype(str)), axis=1).tolist())

# encode categorical features & sequence padding
# using LabelEncoder for simplicity
label_encoders = {}
for feature in ['item_type', 'product_form', 'category', 'sub_category']:
    le = LabelEncoder()
    merged_data[feature] = le.fit_transform(merged_data[feature])
    label_encoders[feature] = le

user_sequences_encoded = user_sequences.apply(lambda x: [label_encoders[feature].transform([item.split('_')[idx] for feature, idx in zip(['item_type', 'product_form', 'category', 'sub_category'], range(4))]) for item in x])
user_sequences_padded = pad_sequences(user_sequences_encoded)


from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
#####






