#Script to create an adjacency matrix/edge list from parent->child pickle file
def create_adj_mat(children):
	fout=open('edge_list.csv','w')
	fout.write('Source;Target\n')
	for key in children.keys():
		p=key
		child1=children[p][0]
		child2=children[p][1]
		if child1!=None:
			fout.write(str(p)+';'+str(child1[0])+'\n')
		if child2!=None:
			fout.write(str(p)+';'+str(child2[0])+'\n')
	fout.close()
if __name__ == '__main__':
	import pickle
	children=pickle.load(open('children.p','rb'))
	create_adj_mat(children)