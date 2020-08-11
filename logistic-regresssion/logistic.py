from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

train_start_index = 0
train_end_index = int(x.shape[0]*0.8)
x_train = x[train_start_index:train_end_index]
y_train = y[train_start_index:train_end_index]

x_valid = x[train_end_index:, :]
y_valid = y[train_end_index:]


clf = LogisticRegression().fit(x_train,y_train)

print(f"Training score {clf.score(x_train,y_train)}")
print(f"valid score {clf.score(x_valid,y_valid)}")

