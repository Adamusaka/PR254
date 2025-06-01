# %% [markdown]
# # Housing Prices Competition for Kaggle Learn Users, grajenje modelov

# %% [markdown]
# # Uvod
# 
# V tej skripti se ukvarjamo z regresijskim problemom napovedovanja cen na podlagi različnih vhodnih značilk, kot je bil predstavljen v Kaggle izzivu. Za gradnjo modelov uporabljamo popularne knjižnice, kot so scikit-learn, XGBoost, CatBoost in LightGBM. Namen je preizkusiti različne regresijske pristope – od preprostih linearnih modelov do kompleksnejših ansambelskih metod, kot sta VotingRegressor in StackingRegressor, s ciljem izboljšati natančnost napovedi in zmanjšati napake. Posebno pozornost namenjamo tudi pripravi in čiščenju podatkov ter odstranjevanju odvečnih vhodnih spremenljivk, da optimiziramo rezultate modelov.

# %%
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

# %%
train_df = pd.read_csv("./Podatki/train.csv")
test_df = pd.read_csv("./Podatki/test.csv")

# %% [markdown]
# # Čiščenje podatkov
# 
# V tem delu kode smo odstranili osamelce in nadomestili manjkajoče vrednosti (večinoma s 'No'), saj v nekaterih primerih vrednost NaN pomeni, da določena lastnost pri objektu ni prisotna, ne pa da podatka dejansko ni ali je neznan. Počistili smo pa tudi podatke v atributih, ki imajo visoko koleracijo.

# %%
plt.scatter(x='LotFrontage', y='SalePrice', data=train_df)
# LotFrontage: Linear feet of street connected to property

# %%
train_df.query('LotFrontage > 300')
# Tukaj smo pregledali nekatere pomembne atribute kot LotArea, LotFrontage in izbrisal najdene osmalece
# 935, 1299, 250, 314, 336, 707, 
# morda: 1397.

# %%
z_scores = pd.Series(stats.zscore(train_df['LotArea']))
z_scores.sort_values(ascending=False).head(10)
# iskanje z scora (za koliko std je daleč od povprečja)

# %%
plt.scatter(x='OverallQual', y='SalePrice', data=train_df)

# %%
plt.scatter(x='OverallCond', y='SalePrice', data=train_df)
train_df.query('OverallCond == 6 & SalePrice > 700000')
# ni tako pomemben atribut
# 379, 1183, 692

# %%
plt.scatter(x='YearBuilt', y='SalePrice', data=train_df)
train_df.query('YearBuilt < 1900 & SalePrice > 400000')
# 186

# %%
# Pri odstranjevanju osamelcev je treba biti previden, saj nekateri izmed njih lahko vsebujejo pomembne informacije, ki so koristne za modeliranje
values = [598, 955, 935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186, 441, 186, 524, 739, 598, 955, 636, 1062, 1191, 496, 198, 1338]

# %%
# Odstranimo zapisane osamelce
train_df = train_df[train_df.Id.isin(values) == False]

# %%
# Pregled null vrednosti
train_df_arranged = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False))
train_df_arranged.head(20)
# Te lahko nadomestimo s povprečnimi vrednostimi ali jih odstranimo.

# %%
objectAtrib = ['FireplaceQu', 'MasVnrType', 'Fence', 'Alley', 'GarageCond', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtQual', 'BsmtCond']
# BsmtExposure
intAtrib = ['MasVnrArea', 'LotFrontage']
unfAtrib = ['BsmtFinType1', 'BsmtFinType2']

for atrib in objectAtrib:
    train_df[atrib].fillna('No', inplace=True)
    test_df[atrib].fillna('No', inplace=True)

for atrib in intAtrib:
    train_df[atrib].fillna(0, inplace=True)
    test_df[atrib].fillna(0, inplace=True)

for atrib in unfAtrib:
    # Unf - unfinished
    train_df[atrib].fillna('Unf', inplace=True)
    test_df[atrib].fillna('Unf', inplace=True)

# SBrkr	Standard Circuit Breakers & Romex
train_df['Electrical'].fillna('SBrkr', inplace=True)
test_df['Electrical'].fillna('SBrkr', inplace=True)


# %%
#Blok kode samo za testiranje
sns.catplot(data=train_df, x="GarageType", y="SalePrice", kind="box")
print(train_df['GarageType'].unique())
# TA -typicall

# %%
train_df = train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])
test_df = test_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])
# Fence, Alley, garage atributi, BsmFinType

# %% [markdown]
# # Feature engineering
# 
# V tem delu smo skupaj ustvarjali različne nove funkcije (feature engineering) in jih testirali. Cilj je bil odstraniti odvečne vhodne spremenljivke ter optimizirati nabor atributov za čim boljši rezultat modela. Pri poimenovanju novih funkcij smo prve črke namenoma zapisali z malo začetnico, da jih lažje prepoznamo kot dodane med procesom.

# %%
train_df['houseAge'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['houseAge'] = test_df['YrSold'] - test_df['YearBuilt']

# %%
train_df['houseRemodelAge'] = train_df['YrSold'] - train_df['YearRemodAdd']
test_df['houseRemodelAge'] = test_df['YrSold'] - test_df['YearRemodAdd']

# %%
train_df['totalSF'] = train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']
test_df['totalSF'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']
# sf - square feet

# %%
train_df['totalArea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
test_df['totalArea'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']

# %%
#Šteilo kopalnic štej 0.5 za samo polovočne kopalnice
train_df['totalBaths'] = train_df['BsmtFullBath'] + train_df['FullBath'] + 0.5 * (train_df['BsmtHalfBath'] + train_df['HalfBath']) 
test_df['totalBaths'] = test_df['BsmtFullBath'] + test_df['FullBath'] + 0.5 * (test_df['BsmtHalfBath'] + test_df['HalfBath']) 

# %%
train_df['totalPorchSF'] = train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch'] + train_df['WoodDeckSF']
test_df['totalPorchSF'] = test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch'] + test_df['WoodDeckSF']

# %%
train_df = train_df.drop(columns=['Id','YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])
test_df = test_df.drop(columns=['YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])

# %%
# korelacija nvega 
correlation_matrix = train_df.corr(numeric_only=True)
plt.figure(figsize=(20,12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# %%
# GarageArea ali GarageCars. Pazi GrageArea ima še nekaj outlierjov (vsaj zgledajo)
train_df = train_df.drop(columns=['GarageArea'])
test_df = test_df.drop(columns=['GarageArea'])

# %%
sns.histplot(train_df, x=train_df['SalePrice'])
# asimetrično

# %% [markdown]
# V prejšnem zagovoru smo omenili, da bomo naredili log transformacijo nad SalePrice. 
# To je transformacija, ki se znebi asimetrije v atributu in zgleda veliko bolje normalno porazdeljena.
# 
# log1p = ln(1 + x)

# %%
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
sns.histplot(train_df, x=train_df['SalePrice'])

# %% [markdown]
# Tukaj bomo vse objektne (kategorijske) spremenljivke pretvorili v ordinalne ali one-hot kodirane spremenljivke, saj regresorjem ne moremo neposredno posredovati neskupinskih (besedilnih) vrednosti

# %%
train_df.dtypes[train_df.dtypes=='object']

# %%
train_df.dtypes[train_df.dtypes!='object']

# %%
# Ordinal - Vrstni red je pomemben
ode_cols = ['LotShape', 'LandContour','Utilities','LandSlope',  'BsmtQual',  'BsmtFinType1',  'CentralAir',  'Functional', \
'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive', 'ExterCond', 'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 'BsmtCond']

# %%
# One hot - Vrstni red ni pomemen
# OneHotEndcoding bo naredil nove columne za vsakega (pretvori v svojo binarni stolpec: 1 če je vrednost in 0 če ni) 
ohe_cols = ['Street', 'LotConfig','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', \
           'MasVnrType','Foundation',  'Electrical',  'SaleType', 'MSZoning', 'SaleCondition', 'Heating', 'GarageType', 'RoofMatl']

# %%
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('SalePrice')

# %%
num_cols

# %%
# Ustvarjam pipeline, da pretvorimo podatke
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

# %%
ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# %%
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# %%
col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols),
    ],
    remainder='passthrough', 
    n_jobs=-1)

# %%
pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)
])

# %%
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# %%
X_preprocessed = pipeline.fit_transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=25)

# %% [markdown]
# # Grajenje modelov
# 
# Uporabili smo preprosto linearno regresijo (LinearRegression), ki predpostavlja linearno povezavo med vhodnimi značilkami in ciljno ceno, ter Ridge regresijo, ki doda L2 regularizacijo za omejevanje velikosti koeficientov. Poleg tega smo preizkusili več metod na drevesni osnovi: Random Forest Regressor (ansambel naključno izbranih dreves), Gradient Boosting Regressor (sekvenčno gradnjo šibkih dreves), XGBoost Regressor (izboljšano izvedbo gradientnega boostanja) in LightGBM Regressor (hitro, ‘leaf-wise’ gradientno boostanje). CatBoost Regressor je samodejno obdelal kategorizirane spremenljivke, medtem ko sta Voting Regressor in Stacking Regressor združila napovedi več modelov z namenom povečati stabilnost in natančnost

# %%
lr = LinearRegression()

# %%
lr.fit(X_train, y_train)

# %%
y_pred_lr = lr.predict(X_test)

# %%
np.sqrt(mean_squared_error(y_test, y_pred_lr))

# %%
RFR = RandomForestRegressor(random_state=13)

# %%
param_grid_RFR = {
    'max_depth': [5, 10, 15],
    'n_estimators': [100, 250, 500],
    'min_samples_split': [3, 5, 10]
}

# %%
rfr_cv = GridSearchCV(RFR, param_grid_RFR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# %%
rfr_cv.fit(X_train, y_train)

# %%
np.sqrt(-1 * rfr_cv.best_score_)

# %%
rfr_cv.best_params_

# %%
XGB = XGBRegressor(random_state=13)

# %%
param_grid_XGB = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [300],
    'max_depth': [3],
    'min_child_weight': [1,2,3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# %%
xgb_cv = GridSearchCV(XGB, param_grid_XGB, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# %%
xgb_cv.fit(X_train, y_train)

# %%
np.sqrt(-1 * xgb_cv.best_score_)

# %%
ridge = Ridge()

# %%
param_grid_ridge = {
    'alpha': [0.05, 0.1, 1, 3, 5, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
}

# %%
ridge_cv = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# %%
ridge_cv.fit(X_train, y_train)

# %%
np.sqrt(-1 * ridge_cv.best_score_)

# %%
GBR = GradientBoostingRegressor()

# %%
param_grid_GBR = {
    'max_depth': [12, 15, 20],
    'n_estimators': [200, 300, 1000],
    'min_samples_leaf': [10, 25, 50],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_features': [0.01, 0.1, 0.7]
}

# %%
GBR_cv = GridSearchCV(GBR, param_grid_GBR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# %%
GBR_cv.fit(X_train, y_train)

# %%
np.sqrt(-1 * GBR_cv.best_score_)

# %%
# Z lgbm regresorjem so bile neke težave zato, ga morda nebomo uporabil v streamlib.
lgbm_regressor = lgb.LGBMRegressor()

# %%
param_grid_lgbm = {
    'boosting_type': ['gbdt'],
    'num_leaves': [20, 31],
    'learning_rate': [0.05],
    'n_estimators': [100, 200]
}

# %%
lgbm_cv = GridSearchCV(lgbm_regressor, param_grid_lgbm, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# %%
lgbm_cv.fit(X_train, y_train)

# %%
np.sqrt(-1 * lgbm_cv.best_score_)

# %%
catboost = CatBoostRegressor(loss_function='RMSE', verbose=False)

# %%
param_grid_cat ={
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.5]
}


# %%
cat_cv = GridSearchCV(catboost, param_grid_cat, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# %%
cat_cv.fit(X_train, y_train)

# %%
np.sqrt(-1 * cat_cv.best_score_)

# %%
vr = VotingRegressor([('gbr', GBR_cv.best_estimator_),
                      ('xgb', xgb_cv.best_estimator_),
                      ('ridge', ridge_cv.best_estimator_)],
                    weights=[2,3,1])

# %%
vr.fit(X_train, y_train)

# %%
y_pred_vr = vr.predict(X_test)

# %%
np.sqrt(mean_squared_error(y_test, y_pred_vr))

# %%
estimators = [
    ('gbr', GBR_cv.best_estimator_),
    ('xgb', xgb_cv.best_estimator_),
    ('cat', cat_cv.best_estimator_),
    #('lgb', lgbm_cv.best_estimator_),
    ('rfr', rfr_cv.best_estimator_),
]

# %%
stackreg = StackingRegressor(
            estimators = estimators,
            final_estimator = vr
)

# %%
stackreg.fit(X_train, y_train)

# %%
y_pred_stack = stackreg.predict(X_test)

# %%
np.sqrt(mean_squared_error(y_test, y_pred_stack))

# %%
df_test_preprocess = pipeline.transform(test_df)

# %%
y_stacking = np.exp(stackreg.predict(df_test_preprocess))

df_y_stacking_out = test_df[['Id']]
df_y_stacking_out['SalePrice'] = y_stacking

df_y_stacking_out.to_csv('submission.csv', index=False)

# %% [markdown]
# Rezultati RMSE na testnem naboru kažejo, da je najboljše napovedi dosegla Ridge regresija z RMSE 0.10905, kar pomeni najmanjšo povprečno absolutno napako v logaritemsko transformirani ciljni spremenljivki. Sledili so Gradient Boosting Regressor (0.11328), Stacking Regressor (0.11373) in CatBoost Regressor (0.11486), ki prav tako zelo dobro ujamejo nelinearne relacije. Voting Regressor (0.11670) je združil sile več modelov, XGBoost (0.11882) in LightGBM (0.12762) sta dosegla solidne, a nekoliko višje napake, medtem ko sta končni mesti zasedla Linear Regression (0.13330) in Random Forest Regressor (0.13388).


