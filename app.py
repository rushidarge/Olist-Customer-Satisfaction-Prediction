import streamlit as st 

# reading data
import pandas as pd
import numpy as np

# Preprocessing
import joblib
from geopy import distance # to find distance between long and lat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# utility libraries
import time
import warnings
import datetime as dt
from tqdm import tqdm
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Model
from sklearn.ensemble import RandomForestClassifier

# feature that we are dropping
drop_feat = ['order_id', 'customer_id', 'order_purchase_timestamp','order_approved_at', \
                 'order_delivered_carrier_date','order_delivered_customer_date', \
                 'order_estimated_delivery_date','customer_unique_id', 'customer_zip_code_prefix', \
                 'customer_city','customer_state', 'order_item_id', 'product_id', 'seller_id',\
                 'shipping_limit_date', 'seller_zip_code_prefix', 'seller_city', 'seller_state',\
                 'seller_geolocation_lat', 'seller_geolocation_lng','customer_geolocation_lat', \
                 'customer_geolocation_lng','order_status']

# features we need to standardize
std_feat = ['payment_value', 'price', 'freight_value', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm',
       'distance', 'purchase_time', 'difference_in_delivery',
       'order_fre_ratio', 'estimate_day', 'actual_delivery', 'frequency',
       'recency', 'monetary', 'age']

# reading seller data
seller_data = pd.read_feather('sellers_dataV2.0.feather')

# readin standard scaler file
std_scale = joblib.load('std_scaler_of_train_dataV1.0.bin')

# load model
randomForest = joblib.load("random_forest_model.joblib")

def Update_feat(input_df):
    """
    Return 
    ----------------
    The dataframe with all preprocessing ready to train or predict for model  
    """
    # making dataframe as a global
    global seller_data
    global std_scale
    # calculate distance
    input_df['distance'] = 0.0
    for id, row in input_df.iterrows():
        cust_ll = tuple(row[['customer_geolocation_lat', 'customer_geolocation_lng']].values)
        seller_ll = tuple(row[['seller_geolocation_lat', 'seller_geolocation_lng']].values)
        dist = distance.distance(cust_ll, seller_ll).km
        input_df.at[id,'distance'] = np.round(dist,3)

    # taking hour from timestamp
    input_df["purchase_time"] = input_df.order_purchase_timestamp.dt.hour
    # taking month from timestamp
    input_df['purchase_month'] = input_df.order_purchase_timestamp.dt.month
    # taking week and weedays from timestamp
    input_df['purchase_weekday'] = input_df.order_purchase_timestamp.dt.weekday
    input_df['purchase_week'] = input_df.order_purchase_timestamp.dt.week

    # to find delivery is late or not
    # estimate - deliver date
    input_df['difference_in_delivery'] = (input_df.order_estimated_delivery_date - input_df.order_delivered_customer_date).dt.days

    def is_late(x):
        if x >= 0:
            # Early
            return 1
        else:
            # late
            return 0
    # Is it late
    input_df['late'] = 0
    input_df.late = input_df.difference_in_delivery.apply(lambda x: is_late(x))
    # order to freight ratio
    # add 1 to avoid divide by 0
    input_df['order_fre_ratio'] = input_df.price / (input_df.freight_value + 1)

    # estimate days in delivery
    input_df['estimate_day'] = (input_df.order_estimated_delivery_date - input_df.order_purchase_timestamp).dt.days

    # actual day to delivery
    input_df['actual_delivery'] = (input_df.order_delivered_customer_date - input_df.order_purchase_timestamp).dt.days


    # one hot encoding of columns payment_type
    coun_vect_pay = CountVectorizer(lowercase=False)
    # to use vocab while deploying
    coun_vect_pay.vocabulary_ = {'boleto': 0, 'credit_card': 1, 'debit_card': 2, 'voucher': 3}
    count_matrix = coun_vect_pay.transform(input_df.payment_type)
    # Converting sparse matrix to array of array
    count_array = count_matrix.toarray()
    # array to dataframe
    one_hot_pay = pd.DataFrame(data=count_array,columns = coun_vect_pay.get_feature_names())
    # Join the encoded input_df
    input_df = pd.concat([input_df,one_hot_pay],axis=1)
    # Drop column payment_type because it is encoded
    input_df = input_df.drop('payment_type', axis=1)

    # we taking only those category who particiapation is in top 20
    # take other categories as other
    prod_eng_ind = ['bed_bath_table', 'health_beauty', 'sports_leisure',
        'computers_accessories', 'furniture_decor', 'housewares',
        'watches_gifts', 'telephony', 'auto', 'toys', 'garden_tools',
        'cool_stuff', 'perfumery', 'baby', 'electronics', 'stationery',
        'fashion_bags_accessories', 'pet_shop', 'consoles_games',
        'office_furniture']
    def cat_val(valu):
        if valu in set(prod_eng_ind):
            return valu
        else:
            return 'other'
    # create new column and map the names
    input_df['cat_english'] = input_df['product_category_name_english']
    input_df['cat_english'] = input_df['cat_english'].apply(lambda x:cat_val(x))

    # one hot encoding of columns category name
    coun_vect_cat = CountVectorizer(lowercase=False)

    # to use vocab while deploying
    coun_vect_cat.vocabulary_ = {'auto': 0, 'baby': 1, 'bed_bath_table': 2, 'computers_accessories': 3, 'consoles_games': 4,\
                                'cool_stuff': 5, 'electronics': 6, 'fashion_bags_accessories': 7, 'furniture_decor': 8, 'garden_tools': 9,\
                                'health_beauty': 10, 'housewares': 11, 'office_furniture': 12, 'other': 13, 'perfumery': 14, 'pet_shop': 15,\
                                'sports_leisure': 16, 'stationery': 17, 'telephony': 18, 'toys': 19, 'watches_gifts': 20}

    count_matrix = coun_vect_cat.transform(input_df.cat_english)
    # Converting sparse matrix to array of array
    count_array = count_matrix.toarray()
    # array to dataframe
    one_hot_cat = pd.DataFrame(data=count_array,columns = coun_vect_cat.get_feature_names())
    # Join the encoded input_df
    input_df = pd.concat([input_df,one_hot_cat],axis=1)
    # Drop column category name because it is encoded
    input_df = input_df.drop(['product_category_name_english','cat_english'], axis=1)

    # Create dictionary of 25 percentile of seller data for those seller who are new to input_df data
    outside_seller = {'frequency':2.0000, 'recency':13.0000, 'monetary':175.7750, 'RFM_Score':2.6975, 'age':123.0000}
    # taking unique seller ids
    input_df_sellers = list(input_df.seller_id.value_counts().index)
    sell_id = list(seller_data.seller_id.value_counts().index)
    # iterate over input_df seller id
    for idd in input_df_sellers:
        # iterate over seller id from seller data if present then do nothing
        if idd in set(sell_id):
            pass
        else:
            # if id is new then we append new id inplace of missing id 
            # and assign 25 percentile values of dataframe
            outside_seller['seller_id'] = idd
            seller_data = seller_data.append(outside_seller, ignore_index=True)
    
    # joining seller data
    input_df = input_df.merge(seller_data, on='seller_id')

    # standard scaler
    input_df.loc[:,std_feat] = std_scale.transform(input_df.loc[:,std_feat])

    # dropping unnecessary features
    input_df.drop(columns=drop_feat,inplace=True)
    return input_df


st.title("Olist Customer Satisfaction Prediction")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Olist Customer Satisfaction Prediction ML App</h2>
</div>
"""

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    
    # Can be used wherever a "file-like" object is accepted:
    X = pd.read_csv(uploaded_file)
    st.write(X)
    
    if type(X) == pd.core.series.Series:
        X = pd.DataFrame(X).T
        X = X.reset_index(drop=True)
    X = X.reset_index(drop=True)
    # Doing preprocessing
    date_col = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',\
            'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date']
    X[date_col] = X[date_col].apply(pd.to_datetime, errors='raise')   
    new_X = Update_feat(X)
    # return prediction output
    if st.button("Predict"):
        result=randomForest.predict(new_X)
        if len(result) == 1:
            if result == 0:
                st.success('Customer is not satisfy')
            else:
                st.success('Customer is satisfy')
        else:
            st.success('Your Output {}'.format(result))
