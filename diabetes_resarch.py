import joblib
import pandas
import seaborn
import catboost
import lightgbm
import xgboost
from matplotlib import pyplot as plot
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pandas.set_option("display.max_columns",None)
pandas.set_option("display.width",500)

df=pandas.read_csv("Datasets/diabetes.csv")

# Exploratary Data Analysis #

def check_dataframe(dataframe,head=5):
    print("########## Shape ##########")
    print(dataframe.shape)
    print("########## Types ##########")
    print(dataframe.dtypes)
    print("########## Head ##########")
    print(dataframe.head(head))
    print("########## Tail ##########")
    print(dataframe.tail(head))
    print("########## NA ##########")
    print(dataframe.isnull().sum())
    print("########## Quantiles ##########")
    print(dataframe.quantile([0,0.05,0.5,0.95,0.99,1]).T)

def cat_summary(dataframe,col_name,plot=False):
    print(pandas.DataFrame({col_name: dataframe[col_name].value_counts(),"Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print(print("###########################"))
    if plot:
        seaborn.countplot(x=dataframe[col_name],data=dataframe)
        plot.show(block=True)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0,0.05,0.5,0.95,0.99,1]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plot.xlabel(numerical_col)
        plot.title(numerical_col)
        plot.show(block=True)

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"},end="\n\n\n"))

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pandas.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}),end="\n\n\n")

def correlation_matrix(dataframe,cols):
        fig = plot.gcf()
        fig.set_size_inches(10,8)
        plot.xticks(fontsize=10)
        plot.yticks(fontsize=10)
        fig=seaborn.heatmap(dataframe[cols].corr(),annot=True,linewidths=0.5,annot_kws={"size":12},linecolor="w",cmap="RdBu")
        plot.show(block=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

check_dataframe(df)

cat_cols,num_cols,cat_but_car=grab_col_names(df,cat_th=5,car_th=20)

for col in cat_cols:
    cat_summary(df,col)

df[num_cols].describe().T

correlation_matrix(df,num_cols)
for col in num_cols:
    target_summary_with_num(df,"Outcome",col)

# Data Preprocessing & Feature Engineering
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pandas.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df.columns=[col.upper() for col in df.columns]

df["NEW_GLUCOSE_CAT"]=pandas.cut(x=df["GLUCOSE"],bins=[-1,139,200],labels=["normal","prediabetes"])

# Age
df.loc[(df["AGE"]<35),"NEW_AGE_CAT"]="young"
df.loc[(df["AGE"]>=35) & (df["AGE"]<=55),"NEW_AGE_CAT"]="middleAge"
df.loc[(df["AGE"]>55),"NEW_AGE_CAT"]="old"

# BMI
df["NEW_BMI_RANGE"]=pandas.cut(x=df["BMI"],bins=[-1, 18.5 , 24.9,29.9,100],labels=["underweight","healty","overweight","obese"])

# BloodPressure
df["NEW_BLOODPRESSURE"]=pandas.cut(x=df["BLOODPRESSURE"],bins=[-1, 79 , 89,123],labels=["normal","hs1","hs2"])

cat_cols,num_cols,cat_but_car=grab_col_names(df,cat_th=5,car_th=20)

for col in cat_cols:
    cat_summary(df,col)

for col in cat_cols:
    target_summary_with_cat(df,"OUTCOME",col)

cat_cols=[col for col in cat_cols if "OUTCOME" not in col]