#Script to get actual decoded rules from raw rules. Here, 2 layer sizes are specified.
#python3 rule_coverage_file2_NN_based.py mode=1/2 acc=100 num_layers=2
import pandas as pd
import pickle
import sys

#mode == 1 if original rules are to be displayed. ==2 if numeric simple rules are to be displayed
mode=sys.argv[1]
# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



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

headers=[]
for item in numeric_features:
    headers.append(item)

for item in categorical_features:
    headers.append(item)
headers.append('UW_DECISION')

df = pd.read_csv("../../data/data_gsp_ordinal_withdecision2.important.without_BR_ID.csv")

def equal_equal(df,feat,val):
    df[feat]=df[feat].apply(lambda x:str(x).strip())
    return df[feat]==val


# In[9]:


def greater_than_equal(df,feat,val):
    df[feat]=df[feat].apply(lambda x:float(x))
    return df[feat]>=val


# In[10]:


def not_equal(df,feat,val):
    df[feat]=df[feat].apply(lambda x:str(x).strip())
    return df[feat]!=val


# In[11]:


def less_than(df,feat,val):
    df[feat]=df[feat].apply(lambda x:float(x))
    return df[feat]<val


# In[12]:


def less_than_equal(df,feat,val):
    df[feat]=df[feat].apply(lambda x:float(x))
    return df[feat]<=val


# In[13]:


def greater_than(df,feat,val):
    print(df[feat])
    df[feat]=df[feat].apply(lambda x:float(x))
    return df[feat]>val


def parse_rule(rule):
    '''
    Parse rule to give dictionary corresponding to
    {'feature':[operator,value]}
    '''
    dictionary={}## key:[operator,value]
    conditions =  rule.split("AND")
    for condition in conditions:
        condition = condition.strip()
        broken_condition =  condition.split(" ")
        if len(broken_condition)==1:
            return False
        key=broken_condition[0]
        if key in dictionary.keys():
            key= key.lower()

        dictionary[key]=[]
        for i in broken_condition[1:]:
            if i!="":
                if len(dictionary[key])==1:
                    if '<' in dictionary[key][0] or '>' in dictionary[key][0]:
                        dictionary[key].append(float(i))
                    else:
                        dictionary[broken_condition[0]].append(i)
                else:
                    dictionary[key].append(str(i).strip())
        if len(dictionary[key])==1:
            return False
        print(dictionary[key])    
        if len(dictionary[key])>2:
            temp_list=[]
            temp_list.append(dictionary[key][0])
            temp_str=""
            for i in dictionary[key][1:]:
                temp_str+=i+" "
            temp_str=temp_str.strip()
            temp_list.append(temp_str)
            dictionary[key]=temp_list
               
    return dictionary
       
           


# In[18]:


def parse_dict(temp_rule_dict,df):
    if temp_rule_dict==False:
        return None,None,None
    series_list=[]

    for key in temp_rule_dict.keys():
        op = temp_rule_dict[key][0]
        
        if key not in ['payor_age','cli_age','proposer_age','age']:
            feat= key.upper()
        else:
            feat=key
        val= temp_rule_dict[key][1]
        

        if op=='==':
            series_list.append(equal_equal(df,feat,val))
        if op=="!=":
            series_list.append(not_equal(df,feat,val))
        if op=='<':
            series_list.append(less_than(df,feat,val))
        if op=='<=':
            series_list.append(less_than_equal(df,feat,val))
        if op=='>':
            series_list.append(greater_than(df,feat,val))
        if op=='=>':
            series_list.append(greater_than_equal(df,feat,val))
   
    sr1=series_list[0]
   
   
    for i in series_list[1:]:
        sr1= (sr1) & (i)

    covered_set=set()
    for index,rows in df[sr1].iterrows():
        covered_set.add(index)
    
    
    examples,positiv_examples  = df[sr1].shape[0],df[sr1]['UW_DECISION'].sum()
    print("covered_set:"+str(covered_set))
    print("examples:"+str(examples))
    return examples,positiv_examples,covered_set




fr = open("../../temp/raw_rules_gsp_2HL_balanced_200MAX.important.without_BR_ID.txt","r")
#forig = open("original_rules_ulip_depth4.txt","r")
rules=[]
for line in fr:
    l=line.rstrip()
    if len(l)>2:
        rules.append(l)

import time
'''
rules_orig=[]
c=0
for line in forig:
    l=line.rstrip()
    if len(l)>2:
        #print("Appending"+str(l))
        #c+=1
        #print(c)
        #print("=======================================")
        #time.sleep(5)
        print(type(l))
        rules_orig.append(l)
'''        
from collections import defaultdict
data_list=[]
acc_dict=defaultdict(list)
#Set that contains df rows satisfying rules with 100% accuracy
coverage_100_acc_set=set()
for i,rule in enumerate(rules):
    rule=rule.replace('  ',' ')
    rule_l=rule.split(' AND ')
    rule_l2=rule_l[0:len(rule_l)-1]
    label_part=rule_l[len(rule_l)-1]
    
    layer1_size=label_part.split(' ')[1]
    if sys.argv[3]=='2':
        layer2_size=label_part.split(' ')[2]
    else:
        layer2_size=None
    #try:
    if ' AND 0.0' in rule:
        lab = 0
    elif ' AND 1.0' in rule:
        lab = 1
    #rule=rule.replace(' AND 0.0','')
    #rule=rule.replace(' AND 1.0','')
    rule=' AND '.join(rule_l2)

    dictionary = parse_rule(rule)
    
    
    
    example,pos_cases,covered_set = parse_dict(dictionary,df)

    mode=int(mode)

    if lab == 1:
        if example==None:
            continue
        acc=(pos_cases/float(example))*100
        if acc>=float(sys.argv[2]):
            coverage_100_acc_set=coverage_100_acc_set.union(covered_set)
        if mode==1: #display original rules
            #print(rules_orig[i])
            #print("+++++++++++++++++++++++++++++++++++")
            data_list.append([rules_orig[i],example,pos_cases,acc,lab,layer1_size,layer2_size])
            acc_dict[acc].append([rules_orig[i],example,pos_cases,'label:',lab,layer1_size,layer2_size])
        elif mode==2:   #display numeric rules
            acc_dict[acc].append([rule,example,pos_cases,'label:',lab,layer1_size,layer2_size])
            data_list.append([rule,example,pos_cases,acc,lab,layer1_size,layer2_size])
    elif lab == 0:
        if example==None:
            continue
        acc=100-((pos_cases/float(example))*100)
        if acc>=float(sys.argv[2]):
            coverage_100_acc_set=coverage_100_acc_set.union(covered_set)
        if mode==1: #display original rules
            #print(rules_orig[i])
            #print("+++++++++++++++++++++++++++++++++++")
            acc_dict[acc].append([rules_orig[i],example,pos_cases,'label:',lab,layer1_size,layer2_size])
            data_list.append([rules_orig[i],example,pos_cases,acc,lab,layer1_size,layer2_size])
        elif mode==2:   #display numeric rules
            acc_dict[acc].append([rule,example,pos_cases,'label:',lab,layer1_size,layer2_size])
            data_list.append([rule,example,pos_cases,acc,lab,layer1_size,layer2_size])
    #except:
        #data_list.append([rule,None,None])
c=0
#for i in range(len(data_list)):
#    c+=1
#    print(data_list[i])
#    if c>20:
#        break
keys=acc_dict.keys()
keys=sorted(keys,reverse=True)
fout=open('../../temp/rules_gsp_balanced_topn_2HL_200MAX.important.without_BR_ID.txt','w')
existing=[]
flag=0


for acc in keys:
    for item in acc_dict[acc]:
        if item not in existing:
            print("Here")
            existing.append(item)
            c+=1
            if float(acc)>=95:
                fout.write("RULE "+str(c)+': '+str(item)+'\t'+str(acc)+'\n\n')
            #if c>20:
            #    flag=1
            #    break
    #if flag==1:
    #    break
print("#Cases covered by "+str(sys.argv[2])+" percent accuracy rules:"+str(len(coverage_100_acc_set)))