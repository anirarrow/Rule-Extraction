from sklearn.datasets import load_boston,load_iris
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import f1_score

import pickle
import pandas as pd
import numpy as np

#GSP 30K*100 features
#numeric_features=['POL_MPREM_AMT', 'TOT_RISK_COVER', 'CVG_MAT_XPRY_DUR', 'CVG_FACE_AMT', 'CVG_SUM_INS_AMT', 'CLI_EARN_INCM_AMT', 'DRUG_CNSM_YR_DUR', 'ALCHL_CNSM_YR_DUR', 'TBCO_CNSM_YR_DUR', 'AVG_ALCHL_QTY', 'PROPOSER_EARN_INCM_AMT', 'PAYOR_EARN_INCM_AMT', 'TRC_PROPOSER', 'TRC_PAYOR', 'cli_age', 'proposer_age', 'payor_age', 'BMI']
#categorical_features=['POL_PRPS_TYP_CD', 'DATA_SRC_CD', 'POL_BILL_MODE_CD', 'PLAN_ID', 'CB_SCORE_LA', 'CVG_STBL_1_CD', 'CVG_STBL_2_CD', 'BR_ID', 'LA_EXST_CLI_IND', 'CLI_BTH_CTRY_CD', 'AGE_PROOF_TYP_CD', 'CLI_SEX_CD', 'CLI_MARIT_STAT_CD', 'CLI_CTZN_CTRY_CD', 'ID_PROOF_TYP_CD', 'CLI_EDUC_TYP_CD', 'CLI_OCCP_TYP_CD', 'OCCP_ID', 'NAT_OF_INDUSTRY', 'CLI_OCCP_RSK_IND', 'CLI_PTL_ACTV_IND', 'CLI_CRIM_OFFNS_IND', 'CLI_SMKR_CD', 'CLI_RSK_RATE_CD', 'CLI_CITY_NM_TXT', 'CLI_CRNT_LOC_CD', 'CLI_CTRY_CD', 'CLI_PSTL_CD', 'CLI_INCM_PROOF_CD', 'GYNCLG_PRBM_IND', 'OTHR_ILL_SURGY_IND', 'NARC_CNSM_IND', 'CLI_NATNLTY_CD', 'CLI_SMK_CIG_IND', 'CLI_DISAB_BNFT_IND', 'CLI_CARDIO_SYS_IND', 'TUMR_CANCER_IND', 'CLI_DIGEST_SYS_IND', 'MUSCL_SKEL_SYS_IND', 'CNSLT_DOCTOR_IND', 'CLI_FEMALE_HLTH_CD', 'URIN_REPRO_SYS_IND', 'CLI_HZRD_AVOC_IND', 'CLI_DIAGNS_TST_IND', 'CLI_AIDS_IND', 'TBCO_CNSM_TYP_CD', 'CLI_DISAB_IND', 'CLI_PHYS_DISAB_CD', 'GLAND_BLOOD_IND', 'CLI_GLAND_DISORD_CD', 'CLI_NERV_SYS_IND', 'CLI_ABSNT_WRK_IND', 'CLI_EENT_DISORD_CD', 'CLI_RESPTY_IND', 'CLI_LIQR_DRINK_IND', 'ALCHL_CNSM_TYP_CD', 'PAYOR_EDUC_TYP_CD', 'IS_LA_PROP_SAME', 'IS_PAYOR_PROP_SAME', 'BNFY2_REL_INSRD_CD', 'PROPOSER_NATNLTY_CD', 'CB_SCORE_PROP', 'BNFY1_REL_INSRD_CD', 'BNFY3_REL_INSRD_CD', 'PROPOSER_SEX_CD', 'CB_SCORE_PAYOR', 'NOMINEE_COUNT', 'PAYOR_BTH_CTRY_CD', 'PROPOSER_RSK_RATE_CD', 'PROPOSER_EDUC_TYP_CD', 'BNFY1_MINR_IND', 'PAYOR_OCCP_TYP_CD', 'PAYOR_SEX_CD', 'BNFY2_MINR_IND', 'PAYOR_MARIT_STAT_CD', 'PAYOR_INCM_PROOF_CD', 'PAYOR_RSK_RATE_CD', 'PAYOR_NATNLTY_CD', 'PROPOSER_BTH_CTRY_CD', 'PAYOR_CTZN_CTRY_CD', 'PROPOSER_MARIT_STAT_CD', 'PROPOSER_OCCP_TYP_CD', 'PROPOSER_CTZN_CTRY_CD', 'PROPOSER_INCM_PROOF_CD']

#ULIP
numeric_features=['POL_MPREM_AMT', 'TOT_RISK_COVER', 'CVG_MAT_XPRY_DUR', 'CVG_FACE_AMT', 'CLI_EARN_INCM_AMT', 'DRUG_CNSM_YR_DUR', 'ALCHL_CNSM_YR_DUR', 'TBCO_CNSM_YR_DUR', 'AVG_ALCHL_QTY', 'PROPOSER_EARN_INCM_AMT', 'PAYOR_EARN_INCM_AMT', 'TRC_PROPOSER', 'TRC_PAYOR', 'cli_age', 'proposer_age', 'payor_age', 'BMI']
categorical_features=['POL_PRPS_TYP_CD', 'DATA_SRC_CD', 'POL_BILL_MODE_CD', 'PLAN_ID', 'CB_SCORE_LA', 'CVG_NUM', 'CVG_STBL_1_CD', 'CVG_STBL_2_CD', 'BR_ID', 'LA_EXST_CLI_IND', 'CLI_BTH_CTRY_CD', 'AGE_PROOF_TYP_CD', 'CLI_SEX_CD', 'CLI_MARIT_STAT_CD', 'CLI_CTZN_CTRY_CD', 'ID_PROOF_TYP_CD', 'CLI_EDUC_TYP_CD', 'CLI_OCCP_TYP_CD', 'OCCP_ID', 'NAT_OF_INDUSTRY', 'CLI_OCCP_RSK_IND', 'CLI_PTL_ACTV_IND', 'CLI_CRIM_OFFNS_IND', 'CLI_SMKR_CD', 'CLI_RSK_RATE_CD', 'CLI_CITY_NM_TXT', 'CLI_CRNT_LOC_CD', 'CLI_CTRY_CD', 'CLI_PSTL_CD', 'CLI_INCM_PROOF_CD', 'GYNCLG_PRBM_IND', 'OTHR_ILL_SURGY_IND', 'NARC_CNSM_IND', 'CLI_NATNLTY_CD', 'CLI_SMK_CIG_IND', 'CLI_DISAB_BNFT_IND', 'CLI_CARDIO_SYS_IND', 'TUMR_CANCER_IND', 'CLI_DIGEST_SYS_IND', 'MUSCL_SKEL_SYS_IND', 'CNSLT_DOCTOR_IND', 'CLI_FEMALE_HLTH_CD', 'URIN_REPRO_SYS_IND', 'CLI_HZRD_AVOC_IND', 'CLI_DIAGNS_TST_IND', 'CLI_AIDS_IND', 'TBCO_CNSM_TYP_CD', 'CLI_DISAB_IND', 'CLI_PHYS_DISAB_CD', 'GLAND_BLOOD_IND', 'CLI_GLAND_DISORD_CD', 'CLI_NERV_SYS_IND', 'CLI_ABSNT_WRK_IND', 'CLI_EENT_DISORD_CD', 'CLI_RESPTY_IND', 'CLI_LIQR_DRINK_IND', 'ALCHL_CNSM_TYP_CD', 'PAYOR_EDUC_TYP_CD', 'IS_LA_PROP_SAME', 'IS_PAYOR_PROP_SAME', 'BNFY2_REL_INSRD_CD', 'PROPOSER_NATNLTY_CD', 'CB_SCORE_PROP', 'BNFY1_REL_INSRD_CD', 'BNFY3_REL_INSRD_CD', 'PROPOSER_SEX_CD', 'CB_SCORE_PAYOR', 'NOMINEE_COUNT', 'PAYOR_BTH_CTRY_CD', 'PROPOSER_RSK_RATE_CD', 'PROPOSER_EDUC_TYP_CD', 'BNFY1_MINR_IND', 'PAYOR_OCCP_TYP_CD', 'PAYOR_SEX_CD', 'BNFY2_MINR_IND', 'PAYOR_MARIT_STAT_CD', 'PAYOR_INCM_PROOF_CD', 'PAYOR_RSK_RATE_CD', 'PAYOR_NATNLTY_CD', 'PROPOSER_BTH_CTRY_CD', 'PAYOR_CTZN_CTRY_CD', 'PROPOSER_MARIT_STAT_CD', 'PROPOSER_OCCP_TYP_CD', 'PROPOSER_CTZN_CTRY_CD', 'PROPOSER_INCM_PROOF_CD', 'CHANNEL']

def get_category_names(feature,closest_value,op,transformed_col,col):
    #For WOE encoding cat_names may contain numeric values as well.
    #For ordinal encoding, it only includes cat variable values.
    cat_names=set()
    if '<=' in op:
        for i in range(0,col.shape[0]):
            if float(transformed_col.iloc[i])>float(closest_value):
                cat_names.add(col.iloc[i])
    elif '>=' in op:
        for i in range(0,col.shape[0]):
            if float(transformed_col.iloc[i])<float(closest_value):
                cat_names.add(col.iloc[i])
    elif '<' in op:
        for i in range(0,col.shape[0]):
            if float(transformed_col.iloc[i])>=float(closest_value):
                cat_names.add(col.iloc[i])
    elif '>' in op:
        for i in range(0,col.shape[0]):
            if float(transformed_col.iloc[i])<=float(closest_value):
                cat_names.add(col.iloc[i])
    
    
    return cat_names

def helper(feature,value,data,x_data,splitter,flag):
    '''
    This method takes the old and transformed dataframes along with the rule's feature and transformed
    value, and returns the original value for that rule.
    value: transformed value for that predicate
    x_data: df containing transformed values
    data: df containing original values
    feature: column name
    ''' 
    
    value=float(value)
    #Convert feature like x_34 to 33, compatible with the x_data headers
    ftr=str(list(data.columns).index(feature))
    #Find the closest value in the transformed data only
    closest_value=float(x_data.iloc[(x_data[ftr]-value).abs().argsort()[:1]][ftr])
    #Find the category names corresponding to the closest value from the original feature file
    if flag=='cat': 
        cat_names=get_category_names(feature,closest_value,splitter,x_data[ftr],data[feature])
        return(cat_names,feature)
    elif flag=='num':
        #Index of closest value in the transformed dataframe (column)
        index_closest_value=x_data.index[x_data[ftr] == closest_value].tolist()[0]
        #The index is used to get the original numeric value from the data
        cat_names=data[feature][index_closest_value]
        return(cat_names,feature)
    #return(data[feature][index_closest_value])
    
    

def extract_original_rule(rule,data,x_data):
    '''
    Method to take a rule with transformed values, and return a rule with the original values
    '''
    #rule_l is the list of predicates for one rule. A predicate is of the form A < B.
    rule_l=rule.split(' AND ')
    operators=['<=','>=','<','>','=','==']    #the order must be maintained like this
    string=''
    for r in rule_l:
        print("PREDICATE:"+str(r))
        for operator in operators:
            if operator in r:
                splitter=' '+operator+' '
                break

        #Get feature, operator, value for one predicate        
        r_l=r.split(splitter)
        feature=r_l[0]

        value=r_l[1]
        #Get original value for this rule with the transformed value for categorical features
        #For WOE, numeric features are also worked upon
        if feature in categorical_features: 
            cat_names,ftr=helper(feature,value,data,x_data,splitter,'cat')
            string+=str(ftr)+' not in '+str(cat_names)+' AND '
        elif feature in numeric_features:
            #COMMENT THIS LINE IF MINMAX NORMALIZATION IS NOT USED
            #cat_names,ftr=helper(feature,value,data,x_data,splitter,'num')
            
            string+=str(feature)+' '+str(splitter)+' '+str(value)+' AND '
    return(string)



target_names = np.array(['rejected','accepted'])
#GSP 30K*100 features
df = pd.read_csv('../ruleex/data/data_ulip_woe_withdecision2.csv')    #x data with UW_Decision

###############################################
#FOR GSP DATA
###############################################
try:
    df=df.drop(['Unnamed: 0'],axis=1)
except:
    print("No column named Unnamed: 0")
##DATA AND X_DATA HAVE ABSOLUTE ONE TO ONE CORRESPONDENCE. HAVE CHECKED. BOTH ALSO START WITH COL '0'.
x_data=df.drop(['UW_DECISION'],axis=1)




#x_data=x_data[feature_names]

#Remove city and postal code
#x_data = x_data.loc[:, x_data.columns != 'CLI_CITY_NM_TXT']
#x_data = x_data.loc[:, x_data.columns != 'CLI_PSTL_CD'] 
#Load y_data from 'data'
y_data=df['UW_DECISION']
X=np.array(x_data)
y=np.array(y_data)
print("X and y shapes are:")
print("x_data")
print(X.shape)
print(y.shape)
#####################################################################################
#TERM
#d=pickle.load(open('./data/data_18k.pkl','rb'))

#GSP 100 features
d=pd.read_csv("../ruleex/data/updated_ULIP_data_final.csv")
try:
    d=d.drop(['Unnamed: 0'],axis=1)
except:
    print("No column named Unnamed: 0")

print("Original data:")
print(d)
print(d.shape)

#exit(0)
df_num=d[numeric_features]
df_cat=d[categorical_features]
data=pd.concat([df_num, df_cat], axis=1)    #data with actual feature names
#data.to_csv("df_18k_organized.csv")    #data where numeric columns are placed before cat columns

#data=pd.read_csv("./data/df_18k_organized.csv")    #data loaded after changing column names
#data.to_csv('./data/data_18k.csv')
print("Original data organized (num columns first):")
#print(data)
print(data.shape)


fin = open('raw_rules_ulip_new.txt','r')
exists=[]
rules=[]
c=0
for line in fin:
    c+=1
    if c%100==0:
        print("Appended "+str(c)+" rules in the list.")
    if line not in exists:    
        exists.append(line.rstrip())
        rules.append((line.rstrip(),1)) #Accept rules only
    else:
        print("Rule exists")
print("List of rules created.")


#rules=['x_34 > 0.4081452190876007 AND x_4 > 0.8599999845027924 AND x_1 > 0.7272727340459824 AND x_24 > 0.37691065669059753 AND x_34 > 0.3425908386707306 AND x_34 > 0.37644776701927185']
frules=open('original_rules_ulip_new.txt','w')
for item in rules:
    line=item[0]
    label=item[1]
    rule=line.rstrip()
    original_rule=extract_original_rule(rule,data,x_data)
    #frules.write(str(rule)+'\n')
    frules.write(original_rule+' '+str(label)+'\n\n')
    print(original_rule)
    #print(data['CLI_CRNT_LOC_CD'])
    print(20*'=')
frules.close()