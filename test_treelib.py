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
			child1=str(children[node][0][0])
			child2=str(children[node][1][0])
			#print(parent_of_node)
			flag=0
			for t in set_of_trees:
				#If the node's parent is present in one of the trees
				if t.contains(parent_of_node):
					#Add the node in that tree
					flag=1
					t.create_node(str(node),str(node), parent=str(parent_of_node))
					#Add its children
					if child1!='None':
						t.create_node(str(child1),str(child1), parent=str(node))
					if child2!='None':
						t.create_node(str(child2),str(child2), parent=str(node))
				elif t.contains(child1):
					flag=1
					t.create_node(str(child1),str(child1), parent=str(node))
				elif t.contains(child2):
					flag=1
					t.create_node(str(child2),str(child2), parent=str(node))
			if flag==0:
				tree=Tree()
				tree.create_node(str(node),str(node))
				print("Added a new non-root node: "+node)


			



			if children[node][0]!=None:
				parent=str(node)
				t=get_tree_for_node(parent,set_of_trees)
				if t!=None:
					t.create_node(str(children[node][0][0]),str(children[node][0][0]), parent=str(parent))
				else:
					t2=get_tree_for_node(str(children[node][0][0]),set_of_trees)
					if t2!=None:
						t2.create_node(str(children[node][0][0]),str(children[node][0][0]), parent=str(parent))

					tree = Tree()
					tree.create_node(str(node),str(node))
					print("Added:"+str(node)+" and created a new tree.")
					tree.create_node(str(children[node][0][0]),str(children[node][0][0]), parent=str(parent))
					print("Added its left child")
					set_of_trees.append(tree)
			if children[node][1]!=None:
				parent=str(node)
				flag=0
				for t in set_of_trees:
					if t.contains(parent):
						flag=1
						break
				if flag==1:
					t.create_node(str(children[node][1][0]),str(children[node][1][0]), parent=str(parent))
				else:
					tree = Tree()
					tree.create_node(str(node),str(node))
					print("Added:"+str(node)+" and created a new tree.")
					tree.create_node(str(children[node][1][0]),str(children[node][1][0]), parent=str(parent))
					print("Added its right child")
					set_of_trees.append(tree)
				#tree.create_node(str(children[node][1][0]),str(children[node][1][0]), parent=str(node))
	print(set_of_trees)
	for t in set_of_trees:
		t.show()

import pickle
children=pickle.load(open('children.p','rb'))
drawTree(children)