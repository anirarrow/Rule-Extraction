#Script to count the number of rules from the rules file (this time the rules file contains the config
#of the neural model)
import sys
fin=open('../../temp/rules_gsp_balanced_2HL_200MAX.important.without_BR_ID.txt','r')
fout=open('../../temp/unique_rules_gsp_balanced_2HL_200MAX.important.without_BR_ID.txt','w')
lol=fin.readlines()

c=0
for line in lol:
	l=line.rstrip()
	print(l)
	if l[0]=='F':
		if c>0:
			print("Writing:"+str(string))
			if string!='':
				fout.write(string+'\n')

		fout.write(l+'\n')
		string=''
		c+=1
	elif l[0]=='r':
		#print('okay')
		print("Appending:"+l)
		string+=l+' && ' 
fout.close()
fin.close()
fin=open('../../temp/unique_rules_gsp_balanced_2HL_200MAX.important.without_BR_ID.txt','r')
fout=open("../../temp/final_rules_gsp_balanced_2HL_200MAX.important.without_BR_ID.txt",'w')
s=set()
for line in fin:
	if line[0]=='F':
		line_part=line.rstrip().split(':')
		label=line_part[1].split(' ')[0]
		layer1_size=line_part[2].split(' ')[0]
		#If there are 2 HLs in the NN
		if sys.argv[1]=='2':
			layer2_size=line_part[3]
	if line[0]!='F' and line[0]=='r':
		print("Adding:"+line.rstrip())
		if sys.argv[1]=='2':
			s.add((line.rstrip(),label,layer1_size,layer2_size))
		elif sys.argv[1]=='1':
			s.add((line.rstrip(),label,layer1_size))

print(s)
#exit(0)
for item in s:
	rule=item[0]
	label=item[1]
	
	layer1_size=item[2]
	if sys.argv[1]=='2':	
		layer2_size=item[3]
		print("WRITING")
		fout.write(rule+' '+str(label)+' '+str(layer1_size)+' '+str(layer2_size)+'\n')
	elif sys.argv[1]=='1':	
		fout.write(rule+' '+str(label)+' '+str(layer1_size)+'\n')
fin.close()
fout.close()