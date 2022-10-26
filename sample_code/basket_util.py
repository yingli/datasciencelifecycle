import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import collections
from math import floor

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


def prep_data(filename, dropna,datecol):
    baskets = pd.read_csv(filename, parse_dates = [datecol])
    if dropna:
        baskets.dropna(inplace=True)
    for s in baskets.columns:
        if ("id" in s):
            baskets.loc[:,s] = baskets.loc[:,s].astype(int)
            #baskets.loc[:,s] = pd.Categorical(baskets.loc[:,s].apply(lambda x: floor(x))) 
            # -- making them into categorical has undesirable effect
    if datecol:
        baskets['date'] = baskets[datecol].dt.date
        baskets['year'] = baskets[datecol].dt.year
        baskets['month'] = baskets[datecol].dt.month
        baskets['month1'] = baskets[datecol].apply(lambda t: t.strftime("%Y-%m"))
        baskets['month_num'] = (baskets['year'] - 2021) * 12 + baskets['month']
        baskets['week_num'] = baskets[datecol].dt.isocalendar().week
        baskets['week_num'] = (baskets['year'] - 2021) * 52 + baskets['week_num']
        baskets['week1'] = baskets[datecol].apply(lambda t: t.strftime("%Y-%W")) # this makes the beginning of Jan 2022 as week 2022-00 , not 2022-52
        baskets.loc[baskets['week_num']==104,'week_num'] = 52
        baskets['day'] = baskets[datecol].dt.day
        baskets['hour'] = baskets[datecol].dt.hour
        baskets['weekday'] = baskets[datecol].dt.weekday
        baskets["spent"] = baskets["qty"] * baskets["price"]
    return baskets

def get_merchant_attributes(baskets):
    merchant_attributes = baskets.groupby(['merchant_id']).agg(
        total_spent = ('spent', 'sum'), 
        num_orders = ('order_id', 'nunique'), 
        first_month = ('month_num', 'min'), 
        last_month = ('month_num', 'max'), 
        num_months = ('month_num', 'nunique'), 
        num_weeks = ('week_num', 'nunique'), 
        num_days = ('date', 'nunique'), 
        num_skus = ('sku_id','nunique'), 
        num_top_cats = ('top_cat_id','nunique'), 
        num_sub_cats = ('sub_cat_id','nunique'),
    ).reset_index()
    merchant_attributes['avg_spent_per_order'] = merchant_attributes.total_spent / merchant_attributes.num_orders
    merchant_attributes['tenure_month'] = merchant_attributes.last_month - merchant_attributes.first_month +1
    return merchant_attributes

def get_order_attributes(baskets):
    order_attributes = baskets.groupby(['order_id']).agg(
        total_spent = ('spent', 'sum'), 
        num_skus = ('sku_id','nunique'), 
        num_top_cats = ('top_cat_id','nunique'), 
        num_sub_cats = ('sub_cat_id','nunique'),
    ).reset_index()
    return order_attributes

def get_sku_attributes(baskets):
    sku_attributes = baskets.groupby(['sku_id']).agg(
        total_spent = ('spent', 'sum'), 
        num_orders = ('order_id', 'nunique'), 
        num_merchants = ('merchant_id', 'nunique'), 
        first_month = ('month_num', 'min'), 
        last_month = ('month_num', 'max'), 
        num_months = ('month_num', 'nunique'), 
        first_week = ('week_num', 'min'), 
        last_week = ('week_num', 'max'), 
        num_weeks = ('week_num', 'nunique'), 
        num_days = ('date', 'nunique'), 
    ).reset_index()
    sku_attributes['avg_spent_per_order'] = sku_attributes.total_spent / sku_attributes.num_orders
    sku_attributes['tenure_month'] = sku_attributes.last_month - sku_attributes.first_month +1
    return sku_attributes

def get_skus_by_day(baskets):
    skus_by_day = baskets.groupby(['sku_id','date']).agg(
        avg_price_by_day = ('price','mean'),
        num_order_by_day = ('order_id', 'nunique'), 
        num_merchants_by_day = ('merchant_id', 'nunique'),
    ).reset_index()
    return skus_by_day

def make_top_cats(baskets):
    top_cats = baskets.groupby(['top_cat_id']).agg(
        avg_price = ('price', 'mean'),
        total_spent = ('spent', 'sum'),
        total_quantity = ('qty' , 'sum'),
        num_orders = ('order_id', 'nunique'), 
        num_days = ('date' , 'nunique'),
        num_merchants = ('merchant_id', 'nunique')
    ).reset_index()
    return top_cats


def run_kmeans_old(df, colnames, k):
    df_for_cluster = df.loc[:,colnames]
    stscaler = StandardScaler().fit(df_for_cluster)
    normalized_df = stscaler.transform(df_for_cluster)
    kmeans = KMeans(init='k-means++',n_clusters=k,n_init=100, max_iter=300, random_state=0).fit(normalized_df)
    df['cluster'] = kmeans.labels_
    df.groupby("cluster").size()
    return df

def run_kmeans(df, colnames, k):
    df_for_cluster = df.loc[:,colnames]
    stscaler = StandardScaler().fit(df_for_cluster)
    normalized_df = stscaler.transform(df_for_cluster)
    kmeans = KMeans(init='k-means++',n_clusters=k,n_init=100, max_iter=300, random_state=0).fit(normalized_df)
    df['cluster'] = kmeans.labels_
    df.groupby("cluster").size()
    return df

def find_elbow(df, colnames, clusters_range):
    df_for_cluster = df.loc[:,colnames]
    stscaler = StandardScaler().fit(df_for_cluster)
    normalized_df = stscaler.transform(df_for_cluster)

    inertias = [] # wcss: Within Cluster Sum of Squares
    for k in clusters_range:
        kmeans = KMeans(init='k-means++',n_clusters=k,n_init=100, max_iter=300, random_state=0).fit(normalized_df)
        inertias.append(kmeans.inertia_)
    plt.figure()
    plt.plot(clusters_range,inertias, marker='o')
    plt.title('Elbow method for deciding on k')
    plt.xlabel('Number of clusters: k')
    plt.ylabel('inertia')
    plt.show()

    return