import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# # read both datasets
# df1 = pd.read_csv('final_integrated_dataset.csv')  # Harry's dataset        
# df2 = pd.read_csv('clean_listings_fromEDA.csv')    # my dataset

# # get column name + dtype for each
# info1 = pd.DataFrame(df1.dtypes, columns=['dtype']).reset_index().rename(columns={'index': 'column'})
# info2 = pd.DataFrame(df2.dtypes, columns=['dtype']).reset_index().rename(columns={'index': 'column'})

# # merge to compare side by side
# compare = info1.merge(info2, on='column', how='outer', suffixes=('_teammate', '_mine'))

# diff = compare[
#     (compare['dtype_mine'] != compare['dtype_teammate']) |
#     compare['dtype_mine'].isna() |
#     compare['dtype_teammate'].isna()
# ]
# print(diff.to_string(index=False))

#print(compare)

# load dataset
df = pd.read_csv("final_integrated_dataset.csv")

print(df.shape)       # 3818*97
df.info()             # quick look at data types & non-null counts

# check % of missing data
missing = (
    (df.isnull().mean().sort_values(ascending=False) * 100)
      .round(2)
)

print(missing)
# monthly_price                          60.27
# security_deposit                       51.13
# weekly_price                           47.38
# cleaning_fee                           26.98
# host_acceptance_rate                   20.25

#####drop monthly_price, security_deposit, and weekly_price

drop_cols = ['monthly_price', 'security_deposit', 'weekly_price']
df = df.drop(columns=drop_cols)

##### input null cleaning_fee with 0, and host_acceptance_rate with median
df['cleaning_fee'] = df['cleaning_fee'].fillna(0)
df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].median())

#identify the 16 object columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(cat_cols)

#['host_name', 'host_since', 'host_response_time', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'bed_type', 'amenities', 'first_review', 'last_review', 'cancellation_policy', 'calendar_date_min', 'calendar_date_max', 'review_first_date', 'review_last_date']

########### transform these 16 columns ##############

# Drop unique text columns
drop_cols = ['host_name', 'amenities']
df = df.drop(columns=drop_cols)

# convert date columns to datetime and derive durations
date_cols = [
    'host_since', 'first_review', 'last_review',
    'calendar_date_min', 'calendar_date_max',
    'review_first_date', 'review_last_date'
]

for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# host tenure (days since joining)
if 'host_since' in df.columns:
    df['host_tenure_days'] = (pd.Timestamp('today') - df['host_since']).dt.days

# review recency
if {'first_review', 'last_review'} <= set(df.columns):
    df['days_since_first_review'] = (pd.Timestamp('today') - df['first_review']).dt.days
    df['days_since_last_review'] = (pd.Timestamp('today') - df['last_review']).dt.days

# calendar span
if {'calendar_date_min', 'calendar_date_max'} <= set(df.columns):
    df['calendar_span_days'] = (df['calendar_date_max'] - df['calendar_date_min']).dt.days

# drop the original date columns
df = df.drop(columns=date_cols, errors='ignore')

# label encode remaining categorical columns ----
cat_cols = [
    'host_response_time', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
    'property_type', 'room_type', 'bed_type', 'cancellation_policy'
]

for c in cat_cols:
    df[c] = df[c].astype(str)
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])

print(df.dtypes.value_counts())

#remove identifiers, and any columns that directly depend on price to avoid data leakage
df = df.drop(columns=['listing_id', 'host_id', 'price_per_person'], errors='ignore')

# ##########Now we are ready to build XGBoost#############
# # create X and y for modeling 

# X = df.drop(columns=['price'])
# y = df['price'].astype(float)

# # Log-transform target
# y_log = np.log1p(y)

# # split the data to training (70%) and test (30%)
# X_train, X_test, y_train_log, y_test_log = train_test_split(
#     X, y_log, test_size=0.3, random_state=42
# )

# # train the XGBoost model
# model = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

# model.fit(X_train, y_train_log)

# # Log-space predictions
# y_pred_log = model.predict(X_test)

# # Back-transform to dollars
# y_test = np.expm1(y_test_log)
# y_pred = np.expm1(y_pred_log)

# # Evaluation (dollar scale)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

# # Also report log-scale RMSE (native model space)
# log_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
# print(f"Log-RMSE: {log_rmse:.3f}")

# # Feature importance
# xgb.plot_importance(model, max_num_features=15)
# plt.title("Top 15 Feature Importances (log-price model)")
# plt.show()

# ########Price reasonableness Checker#########
# df_results = X_test.copy()
# df_results['actual_price'] = y_test
# df_results['pred_price'] = y_pred
# df_results['price_deviation_pct'] = (y_test - y_pred) / y_pred * 100

# def label_reasonableness(pct):
#     if pct > 20:
#         return "Overpriced"
#     elif pct < -20:
#         return "Underpriced"
#     else:
#         return "Reasonable"

# df_results['price_category'] = df_results['price_deviation_pct'].apply(label_reasonableness)

# #print results
# print(df_results[['actual_price', 'pred_price', 'price_deviation_pct', 'price_category']].head(10))

# #check how many listings fall into each group
# print(df_results['price_category'].value_counts())
# # sorted by frequency
# print(df_results['price_category'].value_counts(normalize=True).round(3) * 100)

# #visualize
# sns.countplot(x='price_category', data=df_results, order=['Underpriced', 'Reasonable', 'Overpriced'])
# plt.title('Distribution of Price Reasonableness Categories')
# plt.show()

# ########## deployment#############
# ########## 3-bedroom entire home in Capitol Hill 
# ##########that accommodates 6 guests, has 2 bathrooms, superhost = 1, cleaning fee = $50, etc.

# # Example new listing (must include same features used for training)
# new_listing = {
#     'neighbourhood_cleansed': 'Capitol Hill',
#     'neighbourhood_group_cleansed': 'Central Area',
#     'property_type': 'House',
#     'room_type': 'Entire home/apt',
#     'accommodates': 6,
#     'bedrooms': 3,
#     'bathrooms': 2,
#     'beds': 3,
#     'host_is_superhost': 1,
#     'cleaning_fee': 50,
#     'minimum_nights': 2,
#     'maximum_nights': 90,
#     'number_of_reviews': 25,
#     'review_scores_rating': 4.8,
#     'availability_30': 15,
#     'availability_365': 200,
#     'instant_bookable': 1,
#     # ... and so on for all other features your model expects
# }

# # Convert to DataFrame
# import pandas as pd
# X_new = pd.DataFrame([new_listing])

# # apply same label encoders used during training!
# for c in ['host_response_time', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
#           'property_type', 'room_type', 'bed_type', 'cancellation_policy']:
#     if c in X_new.columns: 
#       X_new[c] = X_new[c].astype(str)
#       X_new[c] = le.fit_transform(X_new[c])  # must use the same encoder used earlier

# # Predict log-price and convert back to $
# pred_log = model.predict(X_new)
# pred_price = np.expm1(pred_log)[0]

# print(f"Predicted reasonable price: ${pred_price:,.2f}")

# =========================
# 1) TRAIN & SAVE ARTIFACTS
# =========================
def train_and_save(df: pd.DataFrame,
                   model_path='xgb_logprice_model.pkl',
                   encoders_path='label_encoders.pkl',
                   features_path='feature_cols.pkl',
                   medians_path='feature_medians.pkl',
                   random_state=42):
    """
    Fits label encoders, trains XGBoost on log(price), evaluates on a 70/30 split,
    saves model + encoders + feature schema + numeric medians.
    Returns (model, X_test, y_test_log, y_pred_log) for quick evaluation/plots.
    """
    # --- Fit & store LabelEncoders for categoricals used in training ---
    cat_cols = [
        'host_response_time', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
        'property_type', 'room_type', 'bed_type', 'cancellation_policy'
    ]
    label_encoders = {}
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            label_encoders[c] = le

    # --- Build X/ y (log target) ---
    X = df.drop(columns=['price'])
    y = df['price'].astype(float)
    y_log = np.log1p(y)

    # Keep training schema & numeric medians (for inference imputation)
    feature_cols = X.columns.tolist()
    feature_medians = X.median(numeric_only=True)

    # --- Split & train ---
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.3, random_state=random_state
    )

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    model.fit(X_train, y_train_log)

    # --- Save artifacts ---
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoders_path)
    joblib.dump(feature_cols, features_path)
    joblib.dump(feature_medians, medians_path)
    print(f"Saved: {model_path}, {encoders_path}, {features_path}, {medians_path}")

    # --- Return for immediate evaluation/plotting ---
    y_pred_log = model.predict(X_test)
    return model, X_test, y_test_log, y_pred_log


# =========================
# 2) EVALUATION HELPERS
# =========================
def evaluate_model(y_test_log, y_pred_log):
    """Prints dollar-scale and log-scale metrics."""
    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    log_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"Log-RMSE: {log_rmse:.3f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Log_RMSE': log_rmse}


def plot_feature_importance(model, title="Top 15 Feature Importances (log-price model)"):
    xgb.plot_importance(model, max_num_features=15)
    plt.title(title)
    plt.show()


def reasonableness_table(X_test, y_test_log, y_pred_log, band=0.20):
    """Returns a DataFrame with actual, predicted, deviation %, and labels."""
    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    df_results = X_test.copy()
    df_results['actual_price'] = y_test
    df_results['pred_price'] = y_pred
    df_results['price_deviation_pct'] = (y_test - y_pred) / y_pred * 100

    def tag(pct):
        if pct > band*100:   return "Overpriced"
        if pct < -band*100:  return "Underpriced"
        return "Reasonable"

    df_results['price_category'] = df_results['price_deviation_pct'].apply(tag)
    return df_results


# =========================
# 3) REUSABLE INFERENCE
# =========================
def load_artifacts(model_path='xgb_logprice_model.pkl',
                   encoders_path='label_encoders.pkl',
                   features_path='feature_cols.pkl',
                   medians_path='feature_medians.pkl'):
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    feature_cols = joblib.load(features_path)
    feature_medians = joblib.load(medians_path)
    return model, label_encoders, feature_cols, feature_medians


def prepare_and_predict(raw: dict,
                        model,
                        label_encoders: dict,
                        feature_cols: list,
                        feature_medians: pd.Series) -> float:
    """
    raw: dict of raw listing fields (strings/numbers) BEFORE encoding
    Returns predicted price in dollars for the single listing
    """
    X_new = pd.DataFrame([raw])

    # Recreate date-derived features if present (must match training logic)
    date_cols = [
        'host_since', 'first_review', 'last_review',
        'calendar_date_min', 'calendar_date_max',
        'review_first_date', 'review_last_date'
    ]
    for c in date_cols:
        if c in X_new.columns:
            X_new[c] = pd.to_datetime(X_new[c], errors='coerce')

    if 'host_since' in X_new.columns:
        X_new['host_tenure_days'] = (pd.Timestamp('today') - X_new['host_since']).dt.days
    if {'first_review','last_review'} <= set(X_new.columns):
        X_new['days_since_first_review'] = (pd.Timestamp('today') - X_new['first_review']).dt.days
        X_new['days_since_last_review']  = (pd.Timestamp('today') - X_new['last_review']).dt.days
    if {'calendar_date_min','calendar_date_max'} <= set(X_new.columns):
        X_new['calendar_span_days'] = (X_new['calendar_date_max'] - X_new['calendar_date_min']).dt.days

    X_new = X_new.drop(columns=date_cols, errors='ignore')

    # Encode categoricals with SAVED encoders
    for c, le in label_encoders.items():
        if c not in X_new.columns:
            X_new[c] = le.classes_[0]  # backfill a known class if missing
        X_new[c] = X_new[c].astype(str).where(
            X_new[c].astype(str).isin(le.classes_), le.classes_[0]
        )
        X_new[c] = le.transform(X_new[c].astype(str))

    # Ensure all training features exist, match order
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = np.nan
    X_new = X_new[feature_cols]

    # Fill numeric NaNs with training medians
    X_new[feature_medians.index] = X_new[feature_medians.index].fillna(feature_medians)

    # Predict log-price -> dollars
    pred_log = model.predict(X_new)[0]
    pred_price = float(np.expm1(pred_log))
    return pred_price


def reasonableness_label(actual_price: float, pred_price: float, band=0.20):
    dev_pct = (actual_price - pred_price) / pred_price * 100
    if dev_pct > band*100:   return "Overpriced",  dev_pct
    if dev_pct < -band*100:  return "Underpriced", dev_pct
    return "Reasonable", dev_pct


# ================
# 4) EXAMPLE USAGE 
# ================
# Train & save
model, X_test, y_test_log, y_pred_log = train_and_save(df)

# Evaluate
metrics = evaluate_model(y_test_log, y_pred_log)
plot_feature_importance(model)

# Reasonableness table + a quick look
df_results = reasonableness_table(X_test, y_test_log, y_pred_log, band=0.20)
print(df_results[['actual_price', 'pred_price', 'price_deviation_pct', 'price_category']].head(10))
print(df_results['price_category'].value_counts())
print((df_results['price_category'].value_counts(normalize=True)*100).round(1))

sns.countplot(x='price_category', data=df_results, order=['Underpriced','Reasonable','Overpriced'])
plt.title('Distribution of Price Reasonableness Categories')
plt.show()

# Load artifacts (simulate a fresh session)
model, label_encoders, feature_cols, feature_medians = load_artifacts()

# Predict a new listing
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
    'instant_bookable': 1,
    # Optional date fields if available:
    # 'host_since': '2018-06-01',
    # 'first_review': '2019-01-15',
    # 'last_review': '2025-09-20',
    # 'calendar_date_min': '2025-01-01',
    # 'calendar_date_max': '2025-12-31',
}

pred = prepare_and_predict(new_listing, model, label_encoders, feature_cols, feature_medians)
print(f"Predicted reasonable price: ${pred:,.2f}")

actual = 400
label, dev = reasonableness_label(actual, pred, band=0.20)
print(f"{label} ({dev:.1f}% vs expected)")
