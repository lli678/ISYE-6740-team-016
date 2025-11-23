#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#handle outliers - wrote a function to exclude datapoints below 0.01 and above 0.99 of population

def winsorize_groupwise(df, group_cols, num_cols, lower_q=0.01, upper_q=0.99):
    
    df_out = df.copy()
    # compute group quantiles
    grouped = df_out.groupby(group_cols)
   
    def clip_series(s):
        # compute group quantiles
        lower = s.quantile(lower_q)
        upper = s.quantile(upper_q)
        return s.clip(lower, upper)
    # apply per-group
    for col in num_cols:
        df_out[col] = grouped[col].transform(lambda s: s.clip(s.quantile(lower_q), s.quantile(upper_q)))
    return df_out

def mape(y_true, y_pred):
   
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


# In[3]:


#load data
df=pd.read_csv('final_integrated_dataset.csv')
print(df.shape)


# In[4]:


#remove $ sign
if df['price'].dtype == object:
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Target
y = df["price"]


# In[5]:


# Features (drop the target)
X = df.drop(columns=["price"])


# In[6]:


#correlation check, so we can exclude variables that are highly correlated with price

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
if len(num_cols) > 0:
    corr = df[num_cols].corrwith(y).abs().sort_values(ascending=False)
    print("\nTop correlated numeric features with price:")
    print(corr.head(10))
else:
    print("\nNo numeric features found for correlation check.")


# In[7]:


# top 6 variables that we can exclude from this analysis as they are highly correlated with price and removing them makes sense as
# they are other forms of price

leakage_cols = [
    'calendar_price_min', 'calendar_price_mean', 'calendar_price_median',
    'calendar_price_max', 'weekly_price', 'monthly_price'
]
X = X.drop(columns=[c for c in leakage_cols if c in X.columns], errors='ignore')


# In[8]:


num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()


# In[9]:


# Ensure neighourhood and room type are there as we use these 2 variables to handle outliers 
group_cols = []
if 'neighbourhood_cleansed' in X.columns:
    group_cols.append('neighbourhood_cleansed')
if 'room_type' in X.columns:
    group_cols.append('room_type')
#sanity check
if len(group_cols) == 0:
   
    X['_single_group'] = 'all'
    group_cols = ['_single_group']
    cat_cols.append('_single_group')


# In[10]:


#using the function to exclude datapoints based on neighbour and room type variables 
X_wins = winsorize_groupwise(pd.concat([X, y], axis=1), group_cols, num_cols, lower_q=0.01, upper_q=0.99)
# separate back
X_wins = X_wins.drop(columns=['price'])


# In[11]:


#scale/standarize the variables by different variable types 

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

# Transform the full dataset 
X_processed = preprocessor.fit_transform(X_wins)


# In[12]:


# get all numerical variables in to an array
num_cols_arr = np.array(num_cols, dtype=object)
# checking cat variables so we can concatenate them later
if len(cat_cols) > 0:
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    encoded_cat_names = encoder.get_feature_names_out(cat_cols)
else:
    encoded_cat_names = np.array([], dtype=object)
feature_names = np.concatenate([num_cols_arr, encoded_cat_names])

# make data frame
X_proc_df = pd.DataFrame(X_processed, columns=feature_names, index=X_wins.index)


# In[13]:


#lasso VS


# In[14]:


y_log = np.log1p(y)   
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_proc_df, y_log)


# In[15]:


coef = pd.Series(lasso.coef_, index=feature_names)
selected_mask = coef.abs() > 1e-6
selected_features = coef[selected_mask].sort_values(key=np.abs, ascending=False).index.tolist()


# In[16]:


print(f"Number of features after LASSO selection: {len(selected_features)}")
print("Selected features (top 30 shown):")
print(selected_features[:30])


# In[45]:


import matplotlib.pyplot as plt


# In[47]:


top6 = selected_features[:6]
plt.figure(figsize=(16, 12))

for i, feature in enumerate(top6, 1):
    plt.subplot(3, 2, i)

    # raw values
    x = X_proc_df[feature]
    y = y  # original price
    
    # scatter plot
    plt.scatter(x, y, alpha=0.3)
    
    plt.xlabel(feature)
    plt.ylabel("Price ($)")
    plt.title(f"Price vs. {feature}")

plt.tight_layout()
plt.show()


# In[17]:


X_selected = X_proc_df[selected_features].copy()
# Split into train/test (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.30, random_state=42)

# We'll model log(price) with OLS (LinearRegression on log1p target)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


# In[18]:


#linear regression on log(price)

model = LinearRegression()
model.fit(X_train, y_train_log)


# In[19]:


# Predictions on test (log space), convert back to price
y_pred_log_test = model.predict(X_test)
y_pred_test = np.expm1(y_pred_log_test)  

# Also predict on full dataset 
y_pred_log_full = model.predict(X_selected)  
y_pred_full = np.expm1(y_pred_log_full)


# In[20]:


results = X_selected.copy(deep=True)
results['actual_price'] = y.values
results['predicted_price'] = y_pred_full
results['price_deviation_pct'] = (np.abs(results['actual_price'] - results['predicted_price']) / 
                                  results['predicted_price']).replace([np.inf, -np.inf], np.nan) * 100


# In[21]:


def assign_tag(row, lower_frac=0.8, upper_frac=1.2):
    a = row['actual_price']
    p = row['predicted_price']
    if p == 0 or pd.isna(p):
        return "Unknown"
    if a < lower_frac * p:
        return "Underpriced"
    elif a > upper_frac * p:
        return "Overpriced"
    else:
        return "Reasonable"


# In[22]:


# Default bands 80/120
results['tag_80_120'] = results.apply(assign_tag, axis=1, lower_frac=0.8, upper_frac=1.2)

# Alternate bands 70/130 and 80/140
results['tag_70_130'] = results.apply(assign_tag, axis=1, lower_frac=0.7, upper_frac=1.3)
results['tag_80_140'] = results.apply(assign_tag, axis=1, lower_frac=0.8, upper_frac=1.4)


# In[23]:


results.head(20)


# In[24]:


#Evaluate the regression on the TEST set
def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape_val, 'R2': r2}

test_metrics = regression_metrics(y_test, y_pred_test)
print("\nTest set metrics (on price scale):")
for k,v in test_metrics.items():
    print(f"  {k}: {v:.4f}")


# In[25]:


# Cross-validation stability (K-Fold CV on the training set)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list, mae_list, mape_list, r2_list = [], [], [], []

X_train_arr = X_train.values
y_train_arr = y_train.values

# cross_val_predict gives out-of-fold predictions in the same order as X_train
oof_pred_log = cross_val_predict(model, X_train, np.log1p(y_train), cv=kf, method='predict')
oof_pred = np.expm1(oof_pred_log)

# Compute overall CV metrics on training
cv_metrics_train = regression_metrics(y_train, oof_pred)
print("\nCross-validated (OOF on training) metrics (price scale):")
for k,v in cv_metrics_train.items():
    print(f"  {k}: {v:.4f}")


# In[ ]:





# In[26]:


print(selected_features)


# In[27]:


def explain_listing_from_selected_vars(row, pred_price, actual_price):
    # ---- reconstruct room type ----
    if row.get("room_type_Entire home/apt", 0) == 1:
        room_type = "Entire home/apt"
    elif row.get("room_type_Shared room", 0) == 1:
        room_type = "Shared room"
    else:
        room_type = "Private room"

    # ---- reconstruct neighborhood group ----
    ng_cols = [
        "neighbourhood_group_cleansed_Downtown",
        "neighbourhood_group_cleansed_Central Area",
        "neighbourhood_group_cleansed_Capitol Hill",
        "neighbourhood_group_cleansed_Queen Anne",
        "neighbourhood_group_cleansed_Cascade"
    ]

    neighbourhood = "Other"
    for col in ng_cols:
        if row.get(col, 0) == 1:
            neighbourhood = col.replace("neighbourhood_group_cleansed_", "")
            break

    # ---- numeric features ----
    bedrooms = row.get("bedrooms", "N/A")
    accommodates = row.get("accommodates", "N/A")
    beds = row.get("beds", "N/A")

    # ---- deviation and tag ----
    deviation_pct = 100 * (actual_price - pred_price) / pred_price

    if deviation_pct < -20:
        tag = "Underpriced"
    elif deviation_pct > 20:
        tag = "Overpriced"
    else:
        tag = "Reasonable"

    explanation = (
        f"{room_type} in {neighbourhood} with {bedrooms:.2f} bedrooms, {beds:.2f} beds, "
        f"and capacity {accommodates:.2f} is expected around ${pred_price:.0f}. "
        f"Your listing is ${actual_price:.0f} ({deviation_pct:+.1f}%). "
        f"Category: {tag}. "
        f"Key drivers include neighborhood group, room type, cleaning fee, "
        f"accommodates, and review scores."
    )

    return explanation


# In[28]:


test_results = pd.DataFrame({
    "actual_price": y_test,
    "predicted_price": y_pred_test
}, index=X_test.index)


# In[29]:


explanations = []

for idx, row in X_test.iterrows():
    explanations.append(
        explain_listing_from_selected_vars(
            row=row,
            pred_price=test_results.loc[idx, "predicted_price"],
            actual_price=test_results.loc[idx, "actual_price"]
        )
    )

test_results["explanation"] = explanations


# In[30]:


test_results['explanation'][1288]


# In[31]:


results.shape


# In[32]:


results['tag_80_120'].eq('Reasonable').sum()


# In[33]:


results['tag_80_120'].eq('Underpriced').sum()


# In[34]:


results['tag_80_120'].eq('Overpriced').sum()


# In[35]:


new_listing = {
    'neighbourhood_cleansed': 'Capitol Hill',
    'neighbourhood_group_cleansed': 'Central Area',
    'property_type': 'House',
    'room_type': 'Entire home/apt',
    'accommodates': 6,
    'bedrooms': 3,
    'bathrooms': 2.0,
    'beds': 3,
    'host_is_superhost': 1,
    'cleaning_fee': 50.0,
    'minimum_nights': 2,
    'maximum_nights': 90,
    'number_of_reviews': 25,
    'review_scores_rating': 4.8,
    'availability_30': 15,
    'availability_60': 30,
    'availability_90': 45,
    'availability_365': 200,
    'instant_bookable': 1
}


# In[ ]:





# In[36]:


new_listing_df = pd.DataFrame([new_listing])


# In[37]:


new_listing_df


# In[38]:


# tmp = pd.concat([new_df, pd.Series([0], name='price')], axis=1)


# In[39]:


# num_cols = X.select_dtypes(include=np.number).columns.tolist()

# ---- FIX: Remove identifier columns ----
# id_like_cols = ["listing_id", "id", "host_id",'host_response_rate','host_acceptance_rate','host_listings_count','host_total_listings_count']
# num_cols = [c for c in num_cols if c in new_listing]


# In[40]:


# 1. Columns used in the trained model
model_features = X_train.columns.tolist()

# 2. Columns actually present in the new listing
new_features = new_listing_df.columns.tolist()

# 3. Intersection (only variables that appear in BOTH)
valid_features = [col for col in model_features if col in new_features]

print("Number of usable features:", len(valid_features))
print("These features will be used:")
print(valid_features)



# In[41]:


# Rebuild the preprocessor using the filtered feature list.
numeric_features = [c for c in valid_features if new_listing_df[c].dtype != 'object']
categorical_features = [c for c in valid_features if new_listing_df[c].dtype == 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)


# In[42]:


X_train_subset = X_train[valid_features]
preprocessor.fit(X_train_subset)


# In[43]:


new_processed = preprocessor.transform(new_listing_df[valid_features])


# In[44]:


predicted_price = model.predict(new_processed)[0]
predicted_price


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




