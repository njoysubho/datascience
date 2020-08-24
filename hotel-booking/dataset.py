import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

PATH = 'data/hotel_bookings.csv'


class DataSet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.c = x.columns

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

    def show(self, size):
        return self.x.head(size)


def get_data(path):
    return pd.read_csv(path)


def from_df(df, split, target):
    train_test_cutoff = int(len(df) * split)
    train_df = hotel_df[0:train_test_cutoff]
    y_train = train_df[target]
    x_train = train_df.drop(columns=[target])

    test_df = hotel_df[train_test_cutoff:]
    y_test = test_df[target]
    x_test = test_df.drop(columns=[target])
    return DataSet(x_train, y_train), DataSet(x_test, y_test)


hotel_df = get_data(PATH)

train_ds, test_ds = from_df(hotel_df, 0.8, 'is_canceled')

print(train_ds.y)

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

model = svm.SVC()

pipeline = Pipeline(steps=[
    ('preprocessors', preprocessors),
    ('model', model)
])

learner = pipeline.fit(train_ds.x, train_ds.y)
print(learner.score())
