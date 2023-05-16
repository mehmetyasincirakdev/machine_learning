import matplotlib.pyplot as plot
import pandas
import seaborn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scikitplot.metrics import plot_roc_curve


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pandas.set_option("display.max_columns", None)
pandas.set_option("display.float_format", lambda x: "%.3f" % x)
pandas.set_option("display.width", 500)

dataFrame = pandas.read_csv("Datasets/diabetes.csv")

dataFrame.head()
dataFrame.shape

dataFrame["Outcome"].value_counts()
seaborn.countplot(x="Outcome", data=dataFrame)
plot.show(block=True)

100 * dataFrame["Outcome"].value_counts() / len(dataFrame)

dataFrame.describe().T
dataFrame["BloodPressure"].hist(bins=20)
plot.xlabel("BloodPressure")
plot.show(block=True)


def plot_numerical_col(dataframe, numeric_col):
    dataframe[numeric_col].hist(bins=20)
    plot.xlabel(numeric_col)
    plot.show(block=True)


for col in dataFrame.columns:
    plot_numerical_col(dataFrame, col)

cols = [col for col in dataFrame.columns if "Outcome" not in col]

dataFrame.groupby("Outcome").agg({"Pregnancies": "mean"})


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(dataFrame, "Outcome", col)

dataFrame.isnull().sum()
dataFrame.describe().T

for col in cols:
    print(col, check_outlier(dataFrame, col))

replace_with_thresholds(dataFrame, "Insulin")

for col in cols:
    dataFrame[col] = RobustScaler().fit_transform(dataFrame[[col]])

dataFrame.head()

y = dataFrame["Outcome"]
X = dataFrame.drop("Outcome", axis=1)
log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)
y_pred[0:10]
y[0:10]


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    seaborn.heatmap(cm, annot=True, fmt=".0f")
    plot.xlabel('y_pred')
    plot.ylabel('y')
    plot.title('Accuracy Score: {0}'.format(acc), size=10)
    plot.show(block=True)


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

plot_roc_curve(log_model, X_test, y_test)
plot.title("ROC Curve")
plot.plot([0, 1], [0, 1], 'r--')
plot.show(block=True)

roc_auc_score(y_test, y_prob)
