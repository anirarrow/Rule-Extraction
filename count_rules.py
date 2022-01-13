#Script to count the number of rules from the rules file
fin=open('rules_ulip_balanced_2HL_automated.txt','r')
fout=open('unique_rules_ulip_balanced_2HL_automated.txt','w')
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
fin=open('unique_rules_ulip_balanced_2HL_automated.txt','r')
fout=open("final_rules_ulip_balanced_2HL_automated.txt",'w')
s=set()
for line in fin:
	if line[0]=='F':
		label=line.rstrip().split(':')[1]
	if line[0]!='F' and line[0]=='r':
		print("Adding:"+line.rstrip())
		s.add((line.rstrip(),label))

#print(s)
#exit(0)
for item in s:
	rule=item[0]
	label=item[1]
	fout.write(rule+' '+str(label)+'\n')
fin.close()
fout.close()