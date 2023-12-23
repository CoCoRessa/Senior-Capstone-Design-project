import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt

df_2013 = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Enterprise Survey/bgd_2013.csv', encoding='euc-kr')
df_merge = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Enterprise Survey/df_2013(2).csv')
#df_merge.drop(['a6b'],axis=1, inplace=True)

df_common = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Enterprise Survey/공통질문정리_2013_2022.csv', encoding='ANSI')
df_common.drop([165,166, 167, 168], axis=0, inplace=True)
df_econ_2013_up = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Economic census/2013_economic_census_upazila.csv')
df_econ_2013_un = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Economic census/2013_economic_census_2013_union.csv')
df_econ_2013_un.drop(['L4_CODE', 'all_emp','Finan_emp', 'Elec_emp', 'Public_emp', 'Manu_emp', 'Const_emp', 'Hotel_emp',
             'Estate_emp', 'Edu_emp', 'Trans_emp'], axis=1, inplace=True)

df_Boost_increase = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Boost/Boost_increase_FY11-13_L2.csv')
df_Boost_increase_pop = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Boost/Boost_increase_per_pop_FY11-13_L2.csv')
df_Boost_sum = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Boost/Boost_sum_FY11-13_L2.csv')
df_Boost_sum_pop = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Boost/Boost_sum_per_pop_FY11-13_L2_.csv')


df_econ_2013_up = df_econ_2013_up[['L3_CODE','all_company','all_emp','all_pop','edu_lit_15_num', 'edu_ter_15_num',	
                             'hou_own',	'nbi_per_capita','urban_pop','urbanization_rate','econ_active_pop_rate','Female_emp_rate','irre_emp_rate','nbi_company_rate']]


df_econ_increase = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Economic census/2013_2022_growth_merge_upazila.csv')
df_econ_increase = df_econ_increase[['L3_CODE', 'Manu_emp_growth','Elec_emp_growth','Const_emp_growth','Hotel_emp_growth','Trans_emp_growth','Finan_emp_growth'	
                      ,'Estate_emp_growth','Public_emp_growth','Edu_emp_growth']]

df_ntl_increase = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Enterprise Survey/NTL_increase-2013.csv')

df_road_2013 = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/Feature selection/Feature_selection_data/Economic census/road_info_L3.csv')
df_road_2013 = df_road_2013[['L3_CODE', 'FY2013_road_lng_km', 'length_per_road_2013']]

df_merge.drop(['l6', 'd2', 'l1'], axis=1, inplace=True)

df_common = df_common[df_common['process']!='Null']
df1 = df_2013[df_common['Qcode']]
continuous_list=df_common[(df_common['Type']=='Continuous')]['Qcode'].tolist()
nominal_list=df_common[(df_common['Type']=='Nominal')]['Qcode'].tolist()

rest_list=df_common[(df_common['Type']=='Continuous') | (df_common['Type']=='Percent')]['Qcode'].tolist()
#rest_list.remove('l6')
rest_list.append('z1')

econ_list = ['all_company','all_emp','all_pop','edu_lit_15_num', 'edu_ter_15_num',
             'Manu_emp_growth','Elec_emp_growth','Const_emp_growth','Hotel_emp_growth','Trans_emp_growth','Finan_emp_growth',
             'Estate_emp_growth','Public_emp_growth','Edu_emp_growth',
             'hou_own',	'nbi_per_capita','urban_pop','urbanization_rate','econ_active_pop_rate','Female_emp_rate','irre_emp_rate','nbi_company_rate',
             'register_emp','irre_emp_rate_union','Ltd_emp_rate', 'Foreign_emp_rate','large_asset_emp_rate', 'export_emp_rate','retail_emp_rate',
             'whole_emp_rate', 'Finan_emp_rate', 'Elec_emp_rate', 'Public_emp_rate', 'Manu_emp_rate', 'Const_emp_rate', 'Hotel_emp_rate', 'Estate_emp_rate',
             'Edu_emp_rate', 'Trans_emp_rate','hh_elec_t',
             #'Finan_emp', 'Elec_emp', 'Public_emp', 'Manu_emp', 'Const_emp', 'Hotel_emp',
             #'Estate_emp', 'Edu_emp', 'Trans_emp',
             'FY2009_SPC_NTL', 'FY2013_SPC_NTL', 'increase_NTL','FY2013_road_lng_km', 'length_per_road_2013']

boost1_col = df_Boost_increase.columns.tolist()
boost1_col.remove('L2_CODE')
boost2_col = df_Boost_increase_pop.columns.tolist()
boost2_col.remove('L2_CODE')
boost3_col = df_Boost_sum.columns.tolist()
boost3_col.remove('L2_CODE')
boost4_col = df_Boost_sum_pop.columns.tolist()
boost4_col.remove('L2_CODE')

boost_total_col = boost1_col+boost2_col+boost3_col+boost4_col

for i in econ_list:
    rest_list.append(i)
#rest_list.append('l1')

for v in boost_total_col:
    rest_list.append(v)

column_list=df1.columns

df_merge_rev = pd.merge(df_merge, df1, how='left',left_on='idstd', right_on='idstd')

for column in df_merge_rev.columns:
    df_merge_rev[column] = np.where(
        ((df_merge_rev[column] == -9) | (df_merge_rev[column] == -8) | (df_merge_rev[column] == -7) | (df_merge_rev[column] == -6) |  (df_merge_rev[column] == -5) |  (df_merge_rev[column] == -4) |
         (df_merge_rev[column] == '-9') | (df_merge_rev[column] == '-8') | (df_merge_rev[column] == '-7') | (df_merge_rev[column] == '-6') |  (df_merge_rev[column] == '-5') |  (df_merge_rev[column] == '-4')),
        np.nan,
        df_merge_rev[column]
    )

obstacle_list = ['c30a', 'd30a', 'd30b', 'SARd31e', 'SARd31f', 'e30', 'g30a', 'i30', 'k30', 'j30a', 'j30b', 'j30c', 'j30e', 'j30f', 'h30', 'l30a', 'l30b']
for j in obstacle_list:
    df_merge_rev[j].replace({2:1, 3:1, 4:1}, inplace=True)

agree_list = ['h7a']
for a in agree_list:
    df_merge_rev[a].replace({'Strongly disagree': 0, 'Tend to disagree':0, 'Tend to agree':1, 'Strongly agree':1}, inplace=True)

drop_list = ['L1_CODE', 'L2_CODE','L2_NAME', 'L3_CODE', 'L3_NAME','d2', 'l1', 'z1_log', 'x1', 'l6', 'Rank','a2','a3a','a4a','a6a',
             'idstd', 'lat_mask', 'lon_mask', 'a6a', 'a0' , 'l5a', 'l5b', 'b6', 'b2d','b6', 'l3a', 'l3b']

nominal_list = [item for item in nominal_list if item not in obstacle_list]
nominal_list = [item for item in nominal_list if item not in agree_list]

# A 리스트에서 B 리스트에 속하는 원소들을 제거합니다.
#nominal_list = remove_elements_in_list(nominal_list, obstacle_list)
#nominal_list.remove('a6b')
nominal_list.append('L1_NAME')
#nominal_list.remove('h1')
#nominal_list.remove('h4a')
nominal_list.append('a4b')
nominal_list.remove('a2')
nominal_list.remove('a3a')
nominal_list.remove('a4a')
nominal_list.remove('a6a')
nominal_list.remove('h5')
nominal_list.append('a6b')

df_merge_rev['a4b'].replace({'Electronics / Electrical': 'Others', 'Leather':'Others', 'Machinery & Equipment': 'Others', 'Chemicals': 'Others', 'Construction & Transport': 'Others',
                            'Non-metallic mineraal products' : 'Others', 'Other manufacturing' : 'Others', 'Furniture' : 'Others', 'Other services' : 'Others'}, inplace=True)
#df_merge_rev['k9'].replace({1:1, 2:2, 3:3, 4:3}, inplace=True)

#실질화
df_merge_rev['z1'] = df_merge_rev['z1'] * 0.624

df_merge_rev['l3b'] = df_merge_rev['l3b'] + (df_merge_rev['l3b'] == 0) * 0.00001
df_merge_rev['production_emp_rate'] = df_merge_rev['l3a'] / df_merge_rev['l3b']
rest_list.append('production_emp_rate')

#df_merge_rev['z1_log'] = np.log1p(df_merge_rev['z1'])

df_merge_rev['a6b'].replace({0: 'SME', 1:'SME', 2:'SME', 3:'Large'}, inplace=True)

#df_merge_rev.drop('n2e', axis=1, inplace=True)
cost_variables = [var for var in df_merge_rev.columns if var.startswith('n')]
cost_variables.append('i2b')

for var in cost_variables:
    df_merge_rev[var] = df_merge_rev[var] / df_merge_rev['x1']

#df_merge_rev['h_innovation'] = df_merge_rev['h1'] + df_merge_rev['h4a']
#df_merge_rev['h_innovation'].replace({0: np.nan, 1:1, 2:1, 3:1, 4:0}, inplace=True)
#nominal_list.append('h_innovation')

df_merge_rev['l_female_rate'] = (df_merge_rev['l5a'] + df_merge_rev['l5b']) / df_merge_rev['x1']
rest_list.append('l_female_rate')

df_merge_rev = pd.merge(df_merge_rev, df_econ_2013_up, how='left', left_on='L3_CODE', right_on='L3_CODE')
df_merge_rev = pd.merge(df_merge_rev, df_econ_2013_un, how='left', left_on='idstd', right_on='idstd')
df_merge_rev = pd.merge(df_merge_rev, df_econ_increase, how='left', left_on='L3_CODE', right_on='L3_CODE')
df_merge_rev = pd.merge(df_merge_rev, df_ntl_increase, how='left', left_on='idstd', right_on='idstd')
df_merge_rev = pd.merge(df_merge_rev, df_road_2013, how='left', left_on='L3_CODE', right_on='L3_CODE')

df_merge_rev = pd.merge(df_merge_rev, df_Boost_increase, how='left', left_on='L2_CODE', right_on='L2_CODE')
df_merge_rev = pd.merge(df_merge_rev, df_Boost_increase_pop, how='left', left_on='L2_CODE', right_on='L2_CODE')
df_merge_rev = pd.merge(df_merge_rev, df_Boost_sum, how='left', left_on='L2_CODE', right_on='L2_CODE')
df_merge_rev = pd.merge(df_merge_rev, df_Boost_sum_pop, how='left', left_on='L2_CODE', right_on='L2_CODE')

#원핫인코딩
df_merge_rev = pd.get_dummies(df_merge_rev, columns = nominal_list)
#df_merge_rev.to_csv('중간수정.csv', encoding='ANSI')
#null값을 가지는 열 개수
null_column_count = df_merge_rev.isnull().sum()[df_merge_rev.isnull().sum() > 0]
print('## Null 피처의 Type :\n', df_merge_rev.dtypes[null_column_count.index])

#값이 object형태인 열
object_columns = list(df_merge_rev.select_dtypes(include=['object']).columns)
print(object_columns)

#왜도값 높은 column ln취하기
from scipy.stats import skew

# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.

skew_features = df_merge_rev[rest_list].apply(lambda x : skew(x))
 #skew 정도가 1 이상인 컬럼들만 추출. 
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))

df_merge_rev[skew_features_top.index] = np.log1p(df_merge_rev[skew_features_top.index])

#피쳐 정규화
from sklearn.preprocessing import StandardScaler

# StandardScaler객체 생성
scaler = StandardScaler()
# StandardScaler 로 데이터 셋 변환. fit( ) 과 transform( ) 호출.
df_merge_rev[rest_list] = scaler.fit_transform(df_merge_rev[rest_list])

print(df_merge_rev)

#학습데이터 테스트 데이터 분리
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#데이터 정리하기
y_target = df_merge_rev['z1']
X_features = df_merge_rev.drop(drop_list,axis=1, inplace=False)
X_features = X_features.drop('z1',axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

X_tr, X_val, y_tr, y_val= train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)

#평가지표 정의
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def get_mae(model, feature_names):
    pred = model.predict(X_test[feature_names])
    mae = mean_absolute_error(y_test, pred)
    print('{0} MAE: {1}'.format(model.__class__.__name__, np.round(mae, 3)))
    return mae

def get_rmse(model, feature_names):
    pred = model.predict(X_test[feature_names])
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    print('{0} RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))
    return rmse

def get_r2score(model, feature_names):
    pred = model.predict(X_test[feature_names])
    r2score = r2_score(y_test, pred) # 함수명을 r2score로 수정
    print('{0} R2_score: {1}'.format(model.__class__.__name__, np.round(r2score, 3))) # 변수명도 r2score로 수정
    return r2score
    
# 여러 모델들을 list 형태로 인자로 받아서 개별 모델들의 RMSE와 R^2을 list로 반환.
def get_maes(models):
    maes =[]
    for model in models:
        mae = get_mae(model)
        maes.append(mae)
    return maes

def get_rmses(models):
    rmses = [ ]
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

def get_r2scores(models):
    r2scores = []
    for model in models:
        r2score = get_r2score(model)
        r2scores.append(r2score)
    return r2scores

#Feature Selection 함수
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from hyperopt import STATUS_OK
from sklearn.model_selection import cross_val_score
import shap

feature_names_to_keep = ['a4b_Retail', 'a4b_Food','a4b_Garments','a4b_Hotel and Restaurants','a4b_Textiles', 'a4b_Others',
                    'L1_NAME_Chittagong', 'L1_NAME_Dhaka', 'L1_NAME_Sylhet','L1_NAME_Rajshahi', 'L1_NAME_Khulna',#'L1_NAME_Barisal',
                    'a6b_Large', 'a6b_SME']
def rmse_expm1(pred, true):
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    return -np.sqrt(np.mean((np.expm1(pred)-np.expm1(true))**2))

def evaluate(x_data, y_data):
    model = XGBRegressor(n_estimators=1000 #,learning_rate=0.02, 
                        #max_depth=14, min_child_weight=11.996,
                        #colsample_bytree=0.787, subsample=0.765,
                        #reg_alpha=4.617, reg_lambda=20.524
                        )
   #X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data, y_data,
                                         #test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set = [(X_tr, y_tr), (X_val, y_val)], early_stopping_rounds=50, verbose=False)
    #val_pred = model.predict(X_test)
    #score = rmse_expm1(val_pred, y_test)
    RMSE = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
    RMSE_average = np.mean(RMSE)
    return RMSE_average

def rfe(x_data, y_data, method, ratio, min_feats=21):
    feats = x_data.columns.tolist()
    archive = pd.DataFrame(columns=['model', 'n_feats', 'feats', 'RMSE', 'R_squared'])
    while True:
        model = XGBRegressor(n_estimators=1000 
                       # max_depth=14, min_child_weight=20.006,
                        #colsample_bytree=0.708, subsample=0.774,
                       # reg_alpha=13.398, reg_lambda=44.359
                        )
        #X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data[feats], y_data,
                                         #test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(x_data[feats], y_data, test_size=0.2, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)
        X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data[feats], y_target,
                                         test_size=0.2, random_state=42)
        fit_params={'early_stopping_rounds': 50,
            'verbose': False,
            'eval_set': [[X_val1, y_val1]]}
        model.fit(X_train, y_train, eval_set = [(X_tr, y_tr), (X_val, y_val)], early_stopping_rounds=50, verbose=False)
        #val_pred = model.predict(X_test)
        #score = rmse_expm1(val_pred, y_test)
        RMSE = -cross_val_score(model, X_tr1, y_tr1, scoring='neg_root_mean_squared_error', cv=5, fit_params = fit_params)
        RMSE_average = round(np.mean(RMSE),3)
        R2score = cross_val_score(model, X_tr1, y_tr1, scoring='r2', cv=5, fit_params = fit_params)
        R2_average = round(np.mean(R2score),3)
        n_feats = len(feats)
        print(n_feats, RMSE_average, R2_average)
        print(feats)
        archive = pd.concat([archive, pd.DataFrame({'model': [model], 'n_feats': [n_feats], 'feats': [feats], 'RMSE': [RMSE_average], 'R_squared': [R2_average]})], ignore_index=True)
        if method == 'basic':
            feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
            feat_imp.drop(feature_names_to_keep,axis=0, inplace=True)
            #print(feat_imp.index)

        elif method == 'perm':
            perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
            feat_imp = pd.Series(perm.feature_importances_, index=feats).sort_values(ascending=False)
            feat_imp.drop(feature_names_to_keep,axis=0, inplace=True)
        elif method == 'shap':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_data[feats])
            feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
            feat_imp.drop(feature_names_to_keep,axis=0, inplace=True)
        next_n_feats = int(len(feat_imp) * ratio)
        print(next_n_feats)
        
        if next_n_feats < min_feats:
            break
        else:
            #feat_imp2 = [feat for feat in feat_imp.index.tolist() if feat not in feature_names_to_keep]
            #feats = feat_imp.iloc[:next_n_feats].index.tolist() + feature_names_to_keep
            #feats = feat_imp2[:next_n_feats] + feature_names_to_keep
            #print(feat_imp.index)
            feats = feat_imp.index.tolist()[:next_n_feats] + feature_names_to_keep
    return archive

#feats = [col for col in df_merge_rev.columns if col != 'z1']
#print(len(feats))
feats = X_features.columns.tolist()
print(len(feats))

for i in [0.9, 0.95, 0.97, 0.99]:
    shap_archive = rfe(X_features, y_target, 'shap', i)
    shap_archive.to_csv('ratio_feature selection %s_2013.csv'%i)
    
for i in [1, 3, 5, 10]:
    shap_archive = rfe(X_features, y_target, 'shap', i)
    shap_archive.to_csv('ratio_feature selection_num %s_2013.csv'%i)
    
##append method 사용
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from hyperopt import STATUS_OK
from sklearn.model_selection import cross_val_score
import shap
    
feature_names_to_keep = ['a4b_Retail', 'a4b_Food','a4b_Garments','a4b_Hotel and Restaurants','a4b_Textiles', 'a4b_Others',
                    'L1_NAME_Chittagong', 'L1_NAME_Dhaka', 'L1_NAME_Sylhet','L1_NAME_Rajshahi', 'L1_NAME_Khulna','L1_NAME_Barisal',
                    'a6b_Large', 'a6b_SME']
def rmse_expm1(pred, true):
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    return -np.sqrt(np.mean((np.expm1(pred)-np.expm1(true))**2))

def evaluate(x_data, y_data):
    model = XGBRegressor(n_estimators=1000 #,learning_rate=0.02, 
                        #max_depth=14, min_child_weight=11.996,
                        #colsample_bytree=0.787, subsample=0.765,
                        #reg_alpha=4.617, reg_lambda=20.524
                        )
   #X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data, y_data,
                                         #test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set = [(X_tr, y_tr), (X_val, y_val)], early_stopping_rounds=50, verbose=False)
    #val_pred = model.predict(X_test)
    #score = rmse_expm1(val_pred, y_test)
    RMSE = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
    RMSE_average = np.mean(RMSE)
    return RMSE_average

def rfe(x_data, y_data, num, max_feats=150):
    
    #feats = feature_names_to_keep
    feats = x_data.columns.tolist()
    selected_feats = feature_names_to_keep
    
    model = XGBRegressor(n_estimators=1000
                       # max_depth=14, min_child_weight=20.006,
                        #colsample_bytree=0.708, subsample=0.774,
                       # reg_alpha=13.398, reg_lambda=44.359
                        )
        #X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data[feats], y_data,
                                         #test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x_data[feats], y_data, test_size=0.2, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)
    X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data[feats], y_target,
                                         test_size=0.2, random_state=42)
    fit_params={'early_stopping_rounds': 50,
            'verbose': False,
            'eval_set': [[X_val1, y_val1]]}
    model.fit(X_train, y_train, eval_set = [(X_tr, y_tr), (X_val, y_val)], early_stopping_rounds=50, verbose=False)
        #val_pred = model.predict(X_test)
        #score = rmse_expm1(val_pred, y_test)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data[feats])
    feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
    #feats = x_data.columns.tolist()
    archive = pd.DataFrame(columns=['model', 'n_feats', 'feats', 'RMSE', 'R_squared'])
    
    while True:
        model = XGBRegressor(n_estimators=1000
                       # max_depth=14, min_child_weight=20.006,
                        #colsample_bytree=0.708, subsample=0.774,
                       # reg_alpha=13.398, reg_lambda=44.359
                        )
        #X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data[feats], y_data,
                                         #test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(x_data[selected_feats], y_data, test_size=0.2, random_state=42)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)
        X_tr1, X_val1, y_tr1, y_val1= train_test_split(x_data[selected_feats], y_target,
                                         test_size=0.2, random_state=42)
        fit_params={'early_stopping_rounds': 50,
            'verbose': False,
            'eval_set': [[X_val1, y_val1]]}
        model.fit(X_train, y_train, eval_set = [(X_tr, y_tr), (X_val, y_val)], early_stopping_rounds=50, verbose=False)
        #val_pred = model.predict(X_test)
        #score = rmse_expm1(val_pred, y_test)
        RMSE = -cross_val_score(model, X_tr1, y_tr1, scoring='neg_root_mean_squared_error', cv=5, fit_params = fit_params)
        RMSE_average = round(np.mean(RMSE),3)
        R2score = cross_val_score(model, X_tr1, y_tr1, scoring='r2', cv=5, fit_params = fit_params)
        R2_average = round(np.mean(R2score),3)
        n_feats = len(selected_feats)
        print(n_feats, RMSE_average, R2_average)
        print(selected_feats)
        archive = pd.concat([archive, pd.DataFrame({'model': [model], 'n_feats': [n_feats], 'feats': [feats], 'RMSE': [RMSE_average], 'R_squared': [R2_average]})], ignore_index=True)
        
        if n_feats > max_feats:
            break
        else:
            #feat_imp2 = [feat for feat in feat_imp.index.tolist() if feat not in feature_names_to_keep]
            #feats = feat_imp.iloc[:next_n_feats].index.tolist() + feature_names_to_keep
            #feats = feat_imp2[:next_n_feats] + feature_names_to_keep
            #print(feat_imp.index)
            #feats = feat_imp.index.tolist()[:next_n_feats] + feature_names_to_keep
            feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
            feat_imp.drop(selected_feats, axis=0, inplace=True)
            selected_feats = feat_imp.index.tolist()[:num] + selected_feats
    return archive
    
for i in [1, 3, 5, 10]:
    shap_archive = rfe(X_features, y_target, i)
    shap_archive.to_csv('append_feature selection_num %s_2013.csv'%i)    