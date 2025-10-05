import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#load data
df = pd.read_csv("listings.csv")
df = df.copy()
#print(df.shape) #(3818, 92)

#check data types
print("\n=== df.info() ===")
df.info()
print("\n=== dtype counts ===")
print(df.dtypes.value_counts())

#check what's in data
#Show all columns
pd.set_option("display.max_columns", None)
#print(df.head(10))
#print(df.columns.tolist())

# check % missing
missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(20))

# ------------------------------
# Data Cleaning Helpers
# ------------------------------

def clean_money(series: pd.Series) -> pd.Series:
    # Handles "$1,234.50" -> 1234.50 and blanks -> NaN
    return (
        series.astype(str)
              .str.replace(r'[,\$]', '', regex=True)
              .replace({'': np.nan, 'nan': np.nan})
              .astype(float)
    )

def clean_percent(series: pd.Series) -> pd.Series:
    # "95%" -> 95.0
    return (
        series.astype(str)
              .str.rstrip('%')
              .replace({'': np.nan, 'nan': np.nan})
              .astype(float)
    )

def yes_no_to_binary(series: pd.Series) -> pd.Series:
    # "t"/"f" or "True"/"False" -> 1/0
    return series.astype(str).str.lower().map({'t':1, 'true':1, 'f':0, 'false':0}).astype('float').fillna(0).astype(int)

# ------------------------------
# Drop columns too sparse / not needed for price reasonableness
# ------------------------------
# Compute missing % to figure out what to drop
missing = df.isnull().mean()

drop_cols = [
    'license','square_feet','weekly_price','monthly_price',
    'listing_url','scrape_id','last_scraped','calendar_last_scraped',
    'thumbnail_url','medium_url','picture_url','xl_picture_url','host_url','host_thumbnail_url','host_picture_url',
    'street','city','state','country_code','country','smart_location','market','neighbourhood_group_cleansed',
    'jurisdiction_names','experiences_offered',
    'summary','space','description','neighborhood_overview','notes','transit','host_about'
]

drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# ------------------------------
# Clean money & percentage columns
# ------------------------------
#Converts money-like strings to floats
money_cols = ['price','weekly_price','monthly_price','security_deposit','cleaning_fee','extra_people']
for c in money_cols:
    if c in df.columns:
        df[c] = clean_money(df[c])

# fees/deposits NaN to 0.0
for c in ['security_deposit','cleaning_fee','extra_people','weekly_price','monthly_price']:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

# Percent to float 
for c in ['host_response_rate','host_acceptance_rate']:
    if c in df.columns:
        df[c] = clean_percent(df[c])

# ------------------------------
# Host & boolean flags
# ------------------------------

# Turn true/false-like strings into 0/1 integers
if 'host_is_superhost' in df.columns:
    df['host_is_superhost'] = yes_no_to_binary(df['host_is_superhost'])

for c in ['has_availability','instant_bookable','require_guest_profile_picture','require_guest_phone_verification',
          'host_has_profile_pic','host_identity_verified','is_location_exact','requires_license']:
    if c in df.columns:
        df[c] = yes_no_to_binary(df[c])

# ------------------------------
# Basic numeric hygiene
# ------------------------------
# coerce to numeric
for c in ['bedrooms','bathrooms','beds','accommodates','availability_30','availability_60','availability_90','availability_365',
          'minimum_nights','maximum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count',
          'review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
          'review_scores_communication','review_scores_location','review_scores_value',
          'latitude','longitude','host_listings_count','host_total_listings_count']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# reviews_per_month NaN -> 0 (no reviews yet)
if 'reviews_per_month' in df.columns:
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)

# # review scores: leave NaN (no reviews)
# review_cols = [c for c in df.columns if c.startswith('review_scores_')]
# for c in review_cols:
#     # optional: median impute
#     med = df[c].median(skipna=True)
#     df[c] = df[c].fillna(med)

# Cap unrealistic nights
if 'minimum_nights' in df.columns:
    df['minimum_nights'] = df['minimum_nights'].clip(lower=1, upper=30)
if 'maximum_nights' in df.columns:
    df['maximum_nights'] = df['maximum_nights'].clip(lower=7, upper=365)

# ------------------------------
# Keep a focused feature set for price modeling
# ------------------------------
keep_cols = [
    # identifiers & location
    'id','neighbourhood_cleansed','latitude','longitude',
    # property & capacity
    'property_type','room_type','accommodates','bedrooms','bathrooms','beds','amenities',
    # pricing
    'price','cleaning_fee','security_deposit','extra_people',
    # host
    'host_id','host_is_superhost','host_response_time','host_response_rate','host_acceptance_rate','host_listings_count',
    # availability & policy
    'availability_30','availability_60','availability_90','availability_365','minimum_nights','maximum_nights',
    'instant_bookable','cancellation_policy',
    # demand signals
    'number_of_reviews','reviews_per_month','review_scores_rating'
]
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()

# ------------------------------
# Create "reasonable price" benchmark and labels
# because price depends heavily on location and room type
# ------------------------------
# baseline: median price by (neighbourhood_cleansed, room_type)
group_keys = [k for k in ['neighbourhood_cleansed','room_type'] if k in df.columns]

if all(k in df.columns for k in group_keys):
    df['median_price'] = df.groupby(group_keys)['price'].transform('median')
else:
    # fallback: median by room_type only, or global median
    if 'room_type' in df.columns:
        df['median_price'] = df.groupby('room_type')['price'].transform('median')
    else:
        df['median_price'] = df['price'].median()

# Avoid division by zero
df['median_price'] = df['median_price'].replace({0: np.nan})

df['price_deviation_pct'] = (df['price'] - df['median_price']) / df['median_price'] * 100

# Classify each listing:
# Overpriced if > +50% above the group median
# Underpriced if < −30% below the group median
# Reasonable otherwise
def label_reasonable(pct):
    # We can tune thresholds after EDA
    if pd.isna(pct):
        return np.nan
    if pct > 50:
        return 'Overpriced'
    elif pct < -30:
        return 'Underpriced'
    else:
        return 'Reasonable'

df['price_category'] = df['price_deviation_pct'].apply(label_reasonable)


# ------------------------------
# Quick sanity checks
# ------------------------------
print("Rows, Cols:", df.shape) #(3818,35)
print(df[['price','median_price','price_deviation_pct','price_category']].head(10))
print("\nMissing % (top 15):\n", df.isnull().mean().sort_values(ascending=False).head(15))

# -----------------------------
# EDA
# -----------------------------
# Price
# 1) Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.show()

# 2) Price deviation categories
sns.countplot(x='price_category', data=df)
plt.title("Price Categories")
plt.show()

# 3) Price vs room type
plt.figure(figsize=(8,5))
sns.boxplot(x='room_type', y='price', data=df)
plt.title("Price by Room Type")
plt.ylim(0, 500)
plt.show()

# 4) Price vs property type
plt.figure(figsize=(12,6))
sns.boxplot(x='property_type', y='price', data=df)
plt.title("Price by Property Type")
plt.xticks(rotation=90)
plt.ylim(0, 500)
plt.show()

# 5) Price vs Accommodates / Bedrooms

plt.figure(figsize=(8,5))
sns.scatterplot(x='accommodates', y='price', data=df, alpha=0.6)
plt.title("Price vs Accommodates")
plt.ylim(0, 500)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title("Price by Bedrooms")
plt.ylim(0, 500)
plt.show()

# Location Effects
# ------------------------------
# Median price per neighborhood
topN = 25
neigh_price = df.groupby('neighbourhood_cleansed')['price'].median().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=neigh_price.values, y=neigh_price.index)
plt.title("Median Price by Neighborhood")
plt.show()

# Map of listings (color by price)
fig = px.scatter_mapbox(
    df, lat="latitude", lon="longitude", color="price", size="accommodates",
    hover_name="neighbourhood_cleansed", hover_data=["room_type","bedrooms"],
    color_continuous_scale="Viridis", zoom=10, height=600
)
fig.update_layout(mapbox_style="carto-positron")
fig.show()

# Map of price categories
fig = px.scatter_mapbox(
    df, lat="latitude", lon="longitude", color="price_category",
    hover_name="neighbourhood_cleansed", hover_data=["room_type","price"],
    zoom=10, height=600
)
fig.update_layout(mapbox_style="carto-positron")
fig.show()

# ------------------------------
# Host Factors
# ------------------------------
#price by superhost
plt.figure(figsize=(6,4))
sns.boxplot(x='host_is_superhost', y='price', data=df)
plt.title("Price by Superhost Status")
plt.ylim(0, 500)
plt.show()

# ------------------------------
# Demand & Reviews???
# ------------------------------


# ------------------------------
# Correlation Heatmap
# ------------------------------

num_cols = ['price','accommodates','bedrooms','beds',
            'number_of_reviews','review_scores_rating','availability_365']

df_corr = df[num_cols].copy()
df_corr['log_price'] = np.log1p(df_corr['price'])

# Pearson
plt.figure(figsize=(9,7))
sns.heatmap(df_corr.corr(method='pearson'), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation (Pearson)")
plt.show()

# Spearman (nonlinear)
plt.figure(figsize=(9,7))
sns.heatmap(df_corr.corr(method='spearman'), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation (Spearman)")
plt.show()
