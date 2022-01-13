#Script to get the number of rules of each type from the TREPAN output

fin=open("rules_german.txt",'r')
c0=0
c1=0
for line in fin:
	if line[0]=='F':
		l=line.rstrip().split(':')
		cl=l[1]
		if cl=='0.0':
			c0+=1
		elif cl=='1.0':
			c1+=1
fin.close()

print("Number of rules for class-0.0 ="+str(c0))
print("Number of rules for class-1.0 ="+str(c1))