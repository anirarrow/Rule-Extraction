import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
import category_encoders as ce
import pickle
from imblearn.over_sampling import SMOTE

#GSP
numeric_features_all = ['POL_MPREM_AMT', 'TOT_RISK_COVER', 'CVG_MAT_XPRY_DUR', 'CVG_FACE_AMT', 'CVG_SUM_INS_AMT', 'CLI_EARN_INCM_AMT', 'DRUG_CNSM_YR_DUR', 'ALCHL_CNSM_YR_DUR', 'TBCO_CNSM_YR_DUR', 'AVG_ALCHL_QTY', 'PROPOSER_EARN_INCM_AMT', 'PAYOR_EARN_INCM_AMT', 'TRC_PROPOSER', 'TRC_PAYOR', 'cli_age', 'proposer_age', 'payor_age', 'BMI']
categorical_features_all =  ['POL_PRPS_TYP_CD', 'DATA_SRC_CD', 'POL_BILL_MODE_CD', 'PLAN_ID', 'CB_SCORE_LA', 'CVG_STBL_1_CD', 'CVG_STBL_2_CD', 'BR_ID', 'LA_EXST_CLI_IND', 'CLI_BTH_CTRY_CD', 'AGE_PROOF_TYP_CD', 'CLI_SEX_CD', 'CLI_MARIT_STAT_CD', 'CLI_CTZN_CTRY_CD', 'ID_PROOF_TYP_CD', 'CLI_EDUC_TYP_CD', 'CLI_OCCP_TYP_CD', 'OCCP_ID', 'NAT_OF_INDUSTRY', 'CLI_OCCP_RSK_IND', 'CLI_PTL_ACTV_IND', 'CLI_CRIM_OFFNS_IND', 'CLI_SMKR_CD', 'CLI_RSK_RATE_CD', 'CLI_CITY_NM_TXT', 'CLI_CRNT_LOC_CD', 'CLI_CTRY_CD', 'CLI_PSTL_CD', 'CLI_INCM_PROOF_CD', 'GYNCLG_PRBM_IND', 'OTHR_ILL_SURGY_IND', 'NARC_CNSM_IND', 'CLI_NATNLTY_CD', 'CLI_SMK_CIG_IND', 'CLI_DISAB_BNFT_IND', 'CLI_CARDIO_SYS_IND', 'TUMR_CANCER_IND', 'CLI_DIGEST_SYS_IND', 'MUSCL_SKEL_SYS_IND', 'CNSLT_DOCTOR_IND', 'CLI_FEMALE_HLTH_CD', 'URIN_REPRO_SYS_IND', 'CLI_HZRD_AVOC_IND', 'CLI_DIAGNS_TST_IND', 'CLI_AIDS_IND', 'TBCO_CNSM_TYP_CD', 'CLI_DISAB_IND', 'CLI_PHYS_DISAB_CD', 'GLAND_BLOOD_IND', 'CLI_GLAND_DISORD_CD', 'CLI_NERV_SYS_IND', 'CLI_ABSNT_WRK_IND', 'CLI_EENT_DISORD_CD', 'CLI_RESPTY_IND', 'CLI_LIQR_DRINK_IND', 'ALCHL_CNSM_TYP_CD', 'PAYOR_EDUC_TYP_CD', 'BNFY2_REL_INSRD_CD', 'PROPOSER_NATNLTY_CD', 'CB_SCORE_PROP', 'BNFY1_REL_INSRD_CD', 'BNFY3_REL_INSRD_CD', 'PROPOSER_SEX_CD', 'CB_SCORE_PAYOR', 'NOMINEE_COUNT', 'PAYOR_BTH_CTRY_CD', 'PROPOSER_RSK_RATE_CD', 'PROPOSER_EDUC_TYP_CD', 'BNFY1_MINR_IND', 'PAYOR_OCCP_TYP_CD', 'PAYOR_SEX_CD', 'BNFY2_MINR_IND', 'PAYOR_MARIT_STAT_CD', 'PAYOR_INCM_PROOF_CD', 'PAYOR_RSK_RATE_CD', 'PAYOR_NATNLTY_CD', 'PROPOSER_BTH_CTRY_CD', 'PAYOR_CTZN_CTRY_CD', 'PROPOSER_MARIT_STAT_CD', 'PROPOSER_OCCP_TYP_CD', 'PROPOSER_CTZN_CTRY_CD', 'PROPOSER_INCM_PROOF_CD']

globally_imp_features=['CLI_PSTL_CD','CLI_CITY_NM_TXT','BMI','OCCP_ID','CVG_FACE_AMT','CLI_CRNT_LOC_CD','POL_MPREM_AMT','TOT_RISK_COVER','NAT_OF_INDUSTRY','cli_age','CLI_EARN_INCM_AMT','proposer_age','CHANNEL','BNFY1_REL_INSRD_CD','AGE_PROOF_TYP_CD','PROPOSER_EARN_INCM_AMT','PROPOSER_EDUC_TYP_CD','ID_PROOF_TYP_CD','CLI_EDUC_TYP_CD','PLAN_ID','CLI_CRIM_OFFNS_IND','CVG_MAT_XPRY_DUR','CLI_INCM_PROOF_CD','payor_age','UW_DECISION']


#Get the important numerical and categorical features
numeric_features=[]
for ftr in numeric_features_all:
    if ftr in globally_imp_features:
        numeric_features.append(ftr)
categorical_features=[]
for ftr in categorical_features_all:
    if ftr in globally_imp_features:
        categorical_features.append(ftr)

'''
Fn to remove unimportant features from the df
'''
def drop_unimportant_features(df):
    unimportant_features=[]
    for col in df.columns:
        if col not in globally_imp_features:
            unimportant_features.append(col)
    df=df.drop(unimportant_features,axis=1)
    return df


'''
Preprocessing fn to convert a df columns to float if possible.
'''
def preprocess_df(df):
    for col in categorical_features:
        col_cleaned_list=[]
        for val in df[col].values:
            try:
                floated_val = float(val)
                col_cleaned_list.append(floated_val)
            except:
                col_cleaned_list.append(val)
        df[col] =  col_cleaned_list

    for col in categorical_features:
        df[col]=df[col].apply(lambda x:str(x).lower().strip())

    for col in numeric_features:
        df[col]=df[col].apply(lambda x:float(x))
    #If only globally important features are to be kept
    if sys.argv[1]=='g':
        df=drop_unimportant_features(df)    
    return df



df=pd.read_csv('../../data/final_gsp_100feat.csv')

print(df.shape)
df=preprocess_df(df)
print(df.shape)
data=df
df.to_csv('../../data/final_gsp_100feat.important.without_BR_ID.csv')

feature_names=df.columns.tolist()






#X = data.drop(['UW_DECISION','POL_ID','CLI_ID'], axis=1)
X = data.drop(['UW_DECISION'], axis=1)
y = data['UW_DECISION']

#Perform SMOTE
le = preprocessing.LabelEncoder()
label_encoder = le.fit(y)
y = label_encoder.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''
encoder_list = [ce.backward_difference.BackwardDifferenceEncoder, 
               ce.basen.BaseNEncoder,
               ce.binary.BinaryEncoder,
                ce.cat_boost.CatBoostEncoder,
                ce.hashing.HashingEncoder,
                ce.helmert.HelmertEncoder,
                ce.james_stein.JamesSteinEncoder,
                ce.one_hot.OneHotEncoder,
                ce.leave_one_out.LeaveOneOutEncoder,
                ce.m_estimate.MEstimateEncoder,
                ce.ordinal.OrdinalEncoder,
                ce.polynomial.PolynomialEncoder,
                ce.sum_coding.SumEncoder,
                ce.target_encoder.TargetEncoder,
                ce.woe.WOEEncoder
                ]

encoder_list=[ce.sum_coding.SumEncoder,
                ce.target_encoder.TargetEncoder,
                ce.woe.WOEEncoder
                ]
'''
#encoder_list=[ce.woe.WOEEncoder]
encoder_list=[ce.ordinal.OrdinalEncoder]                
for encoder in encoder_list:
    
    #numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')),('ordinal', encoder())])
    
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])
    print(X.shape)
    X=X.replace([np.inf, -np.inf], np.nan)
    #print(preprocessor.fit_transform(X_train))
    print(X)
    new_data=preprocessor.fit_transform(X,y)
    new_df=pd.DataFrame(new_data)
    new_df.reset_index()
    print(new_df.shape)

    #Headers in organized form (numeric columns first)
    header_list=[]
    for item in numeric_features:
        header_list.append(item)
    for item in categorical_features:
        header_list.append(item)


    print('CLI_ID' in new_df.columns)
    new_df.to_csv('../../data/data_gsp_ordinal.important.without_BR_ID.csv',index=False,header=header_list)
    #df1=pd.read_csv('../data/data_gsp_woe_balanced.csv')
    #print(df1.shape)
    new_df_with_decision=pd.concat([new_df, pd.DataFrame(y)], axis= 1)
    
    cols=numeric_features

    cols.extend(categorical_features)
    cols.append('UW_DECISION')
    new_df_with_decision.to_csv("../../data/data_gsp_ordinal_withdecision2.important.without_BR_ID.csv",header=cols)
    print(new_df_with_decision.shape)