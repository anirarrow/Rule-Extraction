#Script to get 100% accuracy rules from original rule files
#fin=open("original_rules_ulip_topn_2HL_automated2_98%.txt",'r')
#fin=open("original_rules_ulip_topn_2HL_98.txt",'r')
fin=open("original_rules_ulip_topn_balanced_automated_part.txt",'r')
fout=open("original_trepan_rules.txt",'a')
for line in fin:
	if len(line.split())>0:
		l2=line.rstrip().split(' AND ')
		#l3=l2.split(' ')
		#print(l3)
		l3=l2[len(l2)-1].split(' ')
		
		label=l3[1]
		cov=l3[3]
		pos=l3[5]
		acc=l3[6].replace('\t','')
		if float(acc)==100 and float(cov)>=200:
			fout.write(line+'\n')
fin.close()
fout.close()
	