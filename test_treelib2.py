def get_parent_dict(children):
	from collections import defaultdict
	parents=defaultdict(list)
	for parent in children.keys():
		if children[parent][0]!=None:
			child1=children[parent][0]
			parents[child1[0]].append(parent)
		if children[parent][1]!=None:
			child2=children[parent][1]
			parents[child2[0]].append(parent)
	return parents

def get_tree_for_node(node,set_of_trees):
	for t in set_of_trees:
		if t.contains(node):
			return(t)
	return None

def drawTree(children):
	from treelib import Node, Tree
	parents=get_parent_dict(children)
	print(children)
	print('\n')
	print(parents)
	set_of_trees=[]
	for node in children.keys():
		if node not in parents.keys():
			tree = Tree()
			tree.create_node(str(node),str(node))
			print("Added root:"+str(node)+" and created a new tree.")
			set_of_trees.append(tree)
		else:
			parent_of_node=str(parents[node][0])
			
			if children[node][0]!=None:
				child1=str(children[node][0][0])
			else:
				child1='None'

			if children[node][1]!=None:
				child2=str(children[node][1][0])
			else:
				child2='None'


			#print(parent_of_node)
			flag=0
			for t in set_of_trees:
				#If the node's parent is present in one of the trees
				if t.contains(parent_of_node):
					#Add the node in that tree
					flag=1
					try:
						t.create_node(str(node),str(node), parent=str(parent_of_node))
					except:
						print("Duplicate node error:"+str(node))
					#Add its children
					if child1!='None':
						t2=get_tree_for_node(str(child1),set_of_trees)
						if t2!=None:
							try:
								t2.create_node(str(child1),str(child1), parent=str(node))
							except:
								print("Duplicate node error:"+str(child1))
							#set_of_trees.append(t2)
						else:
							t.create_node(str(child1),str(child1), parent=str(node))
							#set_of_trees.append(t)
						print("Added child node: "+str(child1))
					if child2!='None':
						t2=get_tree_for_node(str(child2),set_of_trees)
						if t2!=None:
							try:
								t2.create_node(str(child2),str(child2), parent=str(node))
							except:
								print("Duplicate node error:"+str(child2))
							#set_of_trees.append(t2)
						else:
							t.create_node(str(child2),str(child2), parent=str(node))
							#set_of_trees.append(t)
						print("Added child node: "+str(child2))
					#except:
					#	print("Duplicate node error:"+str((node,child1,child2)))
					#	continue

				elif t.contains(child1):
					flag=1
					try:
						t.create_node(str(child1),str(child1), parent=str(node))
						print("Added parent "+str(node)+" for child node: "+str(child1))
					except:
						print("Duplicate node error:"+str(node))
					
					if child2!='None':
						try:
							t.create_node(str(child2),str(child2), parent=str(node))
							print("Added parent "+str(node)+" for child node: "+str(child2))
						except:
							print("Parent node not in tree:"+str(node))
						#set_of_trees.append(t)
					#except:
					#	print("Duplicate node error:"+str((node,child1,child2)))
					#	continue
				elif t.contains(child2):
					flag=1
					try:
						t.create_node(str(child2),str(child2), parent=str(node))
						print("Added parent "+str(node)+" for child node: "+str(child2))
					except:
						print("Duplicate node error:"+str(child2))
					if child1!='None':
						try:
							t.create_node(str(child1),str(child1), parent=str(node))
							print("Added parent "+str(node)+" for child node: "+str(child1))
						except:
							print("Parent node not in tree:"+str(node))
						
						#set_of_trees.append(t)
					#except:
					#	print("Duplicate node error:"+str((node,child1,child2)))
					#	continue
			if flag==0:
				tree=Tree()
				tree.create_node(str(node),str(node))
				print("Added a new non-root node: "+str(node))
				if child1!='None':
					tree.create_node(str(child1),str(child1), parent=str(node))
					print("Added its child node: "+str(child1))
				if child2!='None':
					tree.create_node(str(child2),str(child2), parent=str(node))
					print("Added its child node: "+str(child2))
				set_of_trees.append(tree)

	print(set_of_trees)
	for t in set(set_of_trees):
		t.show()

import pickle
children=pickle.load(open('children.p','rb'))
print(children)
drawTree(children)
