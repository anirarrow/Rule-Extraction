#Script to count the number of rules from the rules file
import pickle
import pandas as pd
fin=open('../../temp/final_rules_gsp_balanced_2HL_200MAX.important.without_BR_ID.txt','r')
fout=open('../../temp/raw_rules_gsp_2HL_balanced_200MAX.important.without_BR_ID.txt','w')
#ULIP
#numeric_features=['POL_MPREM_AMT', 'TOT_RISK_COVER', 'CVG_MAT_XPRY_DUR', 'CVG_FACE_AMT', 'CLI_EARN_INCM_AMT', 'DRUG_CNSM_YR_DUR', 'ALCHL_CNSM_YR_DUR', 'TBCO_CNSM_YR_DUR', 'AVG_ALCHL_QTY', 'PROPOSER_EARN_INCM_AMT', 'PAYOR_EARN_INCM_AMT', 'TRC_PROPOSER', 'TRC_PAYOR', 'cli_age', 'proposer_age', 'payor_age', 'BMI']
#categorical_features=['POL_PRPS_TYP_CD', 'DATA_SRC_CD', 'POL_BILL_MODE_CD', 'PLAN_ID', 'CB_SCORE_LA', 'CVG_NUM', 'CVG_STBL_1_CD', 'CVG_STBL_2_CD', 'BR_ID', 'LA_EXST_CLI_IND', 'CLI_BTH_CTRY_CD', 'AGE_PROOF_TYP_CD', 'CLI_SEX_CD', 'CLI_MARIT_STAT_CD', 'CLI_CTZN_CTRY_CD', 'ID_PROOF_TYP_CD', 'CLI_EDUC_TYP_CD', 'CLI_OCCP_TYP_CD', 'OCCP_ID', 'NAT_OF_INDUSTRY', 'CLI_OCCP_RSK_IND', 'CLI_PTL_ACTV_IND', 'CLI_CRIM_OFFNS_IND', 'CLI_SMKR_CD', 'CLI_RSK_RATE_CD', 'CLI_CITY_NM_TXT', 'CLI_CRNT_LOC_CD', 'CLI_CTRY_CD', 'CLI_PSTL_CD', 'CLI_INCM_PROOF_CD', 'GYNCLG_PRBM_IND', 'OTHR_ILL_SURGY_IND', 'NARC_CNSM_IND', 'CLI_NATNLTY_CD', 'CLI_SMK_CIG_IND', 'CLI_DISAB_BNFT_IND', 'CLI_CARDIO_SYS_IND', 'TUMR_CANCER_IND', 'CLI_DIGEST_SYS_IND', 'MUSCL_SKEL_SYS_IND', 'CNSLT_DOCTOR_IND', 'CLI_FEMALE_HLTH_CD', 'URIN_REPRO_SYS_IND', 'CLI_HZRD_AVOC_IND', 'CLI_DIAGNS_TST_IND', 'CLI_AIDS_IND', 'TBCO_CNSM_TYP_CD', 'CLI_DISAB_IND', 'CLI_PHYS_DISAB_CD', 'GLAND_BLOOD_IND', 'CLI_GLAND_DISORD_CD', 'CLI_NERV_SYS_IND', 'CLI_ABSNT_WRK_IND', 'CLI_EENT_DISORD_CD', 'CLI_RESPTY_IND', 'CLI_LIQR_DRINK_IND', 'ALCHL_CNSM_TYP_CD', 'PAYOR_EDUC_TYP_CD', 'IS_LA_PROP_SAME', 'IS_PAYOR_PROP_SAME', 'BNFY2_REL_INSRD_CD', 'PROPOSER_NATNLTY_CD', 'CB_SCORE_PROP', 'BNFY1_REL_INSRD_CD', 'BNFY3_REL_INSRD_CD', 'PROPOSER_SEX_CD', 'CB_SCORE_PAYOR', 'NOMINEE_COUNT', 'PAYOR_BTH_CTRY_CD', 'PROPOSER_RSK_RATE_CD', 'PROPOSER_EDUC_TYP_CD', 'BNFY1_MINR_IND', 'PAYOR_OCCP_TYP_CD', 'PAYOR_SEX_CD', 'BNFY2_MINR_IND', 'PAYOR_MARIT_STAT_CD', 'PAYOR_INCM_PROOF_CD', 'PAYOR_RSK_RATE_CD', 'PAYOR_NATNLTY_CD', 'PROPOSER_BTH_CTRY_CD', 'PAYOR_CTZN_CTRY_CD', 'PROPOSER_MARIT_STAT_CD', 'PROPOSER_OCCP_TYP_CD', 'PROPOSER_CTZN_CTRY_CD', 'PROPOSER_INCM_PROOF_CD', 'CHANNEL']

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

df=pd.read_csv("../../data/final_gsp_100feat.important.without_BR_ID.csv")


numeric_features.extend(categorical_features)

for line in fin:
	line=line.rstrip()
	line=line.replace("rule:[(","")
	line=line.replace(")]","")
	predicates=line.split("&&")
	string=''
	for pred in predicates:
		pred_l=pred.split(",")
		if len(pred_l)==1:
			string+=pred
			break
		print(pred_l)
		#Major change is -1. Needs to be changed in TERM.
		ftr=int(pred_l[0])-1
		print(ftr)
		print(len(numeric_features))
		feature=numeric_features[ftr]
		val=pred_l[2]
		opr=str(pred_l[1].rstrip())
		opr=str(pred_l[1].lstrip())
		
		if "gte" in opr:
			op='>='
		elif "lte" in opr:
			op='<='
		elif "gt" in opr:
			op='>'
		elif "lt" in opr:
			op='<'

		string+=feature+' '+op+' '+str(val)+' AND '
	print(string)
	fout.write(string+'\n\n')
