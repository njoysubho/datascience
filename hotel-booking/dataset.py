import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

PATH = 'data/hotel_bookings.csv'


def get_data(path):
    return pd.read_csv(path)


hotel_df = get_data(PATH)
y = hotel_df['is_canceled']
x = hotel_df.drop(columns=['is_canceled'])

num_features = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                "babies", "is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled", "agent", "company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel", "arrival_date_month", "meal", "market_segment",
                "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]

num_transformer = SimpleImputer(strategy='constant')
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
    ("one-hot", OneHotEncoder(handle_unknown='ignore'))
])

preprocessors = ColumnTransformer(transformers=[
    ("numerical", num_transformer, num_features),
    ("categorical", cat_transformer, cat_features)
])

normalizer = Normalizer()

all_models = [('DecisionTree', tree.DecisionTreeClassifier(random_state=42)),
              ('LogisticRegression', LogisticRegression(random_state=42, max_iter=120)),
              ('SGDClassifier', SGDClassifier(random_state=42))]

y = hotel_df['is_canceled']
x = hotel_df.drop(columns=['is_canceled'])

split = KFold(n_splits=4, shuffle=True, random_state=42)
for name, model in all_models:
    pipeline = Pipeline(steps=[
        ('preprocessors', preprocessors),
        ('normalize', Normalizer()),
        ('model', model)
    ])
    score = cross_val_score(pipeline, x, y, cv=split, verbose=1, scoring='accuracy')
    print(f'Model={name} details=[mean_score={np.mean(score)},min_score={np.min(score)},max_score={np.max(score)}]')
