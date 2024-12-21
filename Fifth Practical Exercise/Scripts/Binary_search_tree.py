# Task 1 - Binary Search Tree

import networkx as nx
import matplotlib.pyplot as plt

# Definition of the Node class
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        
# Definition of the BinarySearchTree class
class BinarySearchTree:
    def __init__(self):
        self.root = None
    
#Method to insert a key into the tree
    def insert(self, key):
        self.root = self._insert(self.root, key)

# Private method for recursive insertion
    def _insert(self, root, key):
        if root is None:
            return Node (key)
        if key < root.key:
            root.left = self._insert(root.left, key)
        else:
            root.right = self._insert(root.right, key)
        return root
    
# Method to search for a key in the tree
    def search(self, key):
        return self._search(self.root, key)

# Private method for recursive search
    def _search(self, root, key):
        if root is None:
            return False
        if root.key == key:
            return True
        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)
    
# Method to delete a key from the tree
    def delete(self, key):
        self.root = self._delete(self.root, key)

# Private method for recursive deletion
    def _delete(self, root, key):
        if root is None:
            return root
        if key < root.key:
            root.left = self._delete(root.left, key)
        elif key > root.key:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            root.key = self._get_min_value(root.right)
            root.right = self._delete(root.right, root.key)
        return root
    
# Private method to get the minimum value in a tree
    def _get_min_value(self, root):
        while root.left is not None:
            root = root.left
        return root.key
    
# Method for inorder traversal of the tree
    def inorder_traversal(self):
        result = []
        self._inorder_traversal(self.root, result)
        return result
    
# Private method for recursive inorder traversal
    def _inorder_traversal(self, root, result):
        if root is not None:
            self._inorder_traversal(root.left, result)
            result.append(root.key)
            self._inorder_traversal(root.right, result)

# Method to visualize the tree
    def plot_tree(self, title, color = "skyblue"):
        G = nx.DiGraph()
        pos = self._build_graph(G, self.root)
        plt.title(title) # put this before drawing the graph because it will be overwritten by the graph otherwise
        nx.draw(G, pos, with_labels=True, arrows=False, node_size=800, node_color=color, font_size=10)
        plt.show()

# Private method to recursively build the graph for visualization
    def _build_graph(self, G, node,pos=None, x=0, y=0, layer=1):
        if pos is None:
            pos = {node.key: (x,y)}
        else:
            pos[node.key] = (x,y)

        if node.left is not None:
            left_pos= (x-1/(2**layer), y-1)
            G.add_edge(node.key, node.left.key)
            self._build_graph(G, node.left, pos, x-1/(2**layer), y-1, layer+1)

        if node.right is not None:
            right_pos = (x+1/(2**layer), y-1)
            G.add_edge(node.key, node.right.key)
            self._build_graph(G, node.right, pos, x+1/(2**layer), y-1, layer+1)
        return pos
    
# Main code

#Tree A
print("Tree A in progress...\n")
a = [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]

#Create a binary search tree
bst_a = BinarySearchTree()

for num in a: 
    bst_a.insert(num)

# Displaying the initial tree graphically
plt.figure(facecolor="pink")
bst_a.plot_tree(title="Tree A", color="pink")

# Adding the value 15
value_to_insert = 15
plt.figure(facecolor="lightblue")
bst_a.insert(value_to_insert)
# Displaying the tree graphically after the insertion
bst_a.plot_tree(title=f"Tree A after inserting {value_to_insert}", color="lightblue")


# Deleting the value 27
value_to_delete = 27
plt.figure(facecolor="lightgreen")
bst_a.delete(value_to_delete)
# Displaying the tree graphically after the deletion
bst_a.plot_tree(title=f"Tree A after deleting {value_to_delete}", color="lightgreen")

# Searching for the value 65
value_to_search = 65
result = bst_a.search(value_to_search)
print(f"Searching for {value_to_search} in Tree A: {result}")

value_to_search = 2
result = bst_a.search(value_to_search)
print(f"Searching for {value_to_search} in Tree A: {result}")

# Displaying the final tree graphically
plt.figure(facecolor="orange")
bst_a.plot_tree(title="Final Tree A", color="orange")
plt.show()
print("Tree A completed.\n")

# Tree B
print("Tree B in progress...\n")
b = [149, 38, 65, 197, 60, 176, 13, 217, 5, 11]
bst_b = BinarySearchTree()

for num in b: 
    bst_b.insert(num)

plt.figure(facecolor="lightgreen")
bst_b.plot_tree(title="Tree B", color="lightgreen")

value_to_insert = 105
plt.figure(facecolor="lightblue")
bst_b.insert(value_to_insert)
bst_b.plot_tree(title=f"Tree B after inserting {value_to_insert}", color="lightblue")

value_to_delete = 217
plt.figure(facecolor="lightgreen")
bst_b.delete(value_to_delete)
bst_b.plot_tree(title=f"Tree B after deleting {value_to_delete}", color="lightgreen")

value_to_search = 10
result = bst_b.search(value_to_search)
print(f"Searching for {value_to_search} in Tree B: {result}")

value_to_search = 5
result = bst_b.search(value_to_search)
print(f"Searching for {value_to_search} in Tree B: {result}")

plt.figure(facecolor="orange")
bst_b.plot_tree(title="Final Tree B", color="orange")
plt.show()
print("Tree B completed.\n")


# Tree C
print("Tree C in progress...\n")
c = [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]
bst_c = BinarySearchTree()
for num in c:
    bst_c.insert(num)
plt.figure(facecolor="lightblue")
bst_c.plot_tree(title="Tree C", color="lightblue")

value_to_insert = 48
plt.figure(facecolor="lightgreen")
bst_c.insert(value_to_insert)
bst_c.plot_tree(title=f"Tree C after inserting {value_to_insert}", color="lightgreen")

value_to_delete = 77
plt.figure(facecolor="lightblue")
bst_c.delete(value_to_delete)
bst_c.plot_tree(title=f"Tree C after deleting {value_to_delete}", color="lightblue")

value_to_search = 55
result = bst_c.search(value_to_search)
print(f"Searching for {value_to_search} in Tree C: {result}")

value_to_search = 100
result = bst_c.search(value_to_search)
print(f"Searching for {value_to_search} in Tree C: {result}")

plt.figure(facecolor="orange")
bst_c.plot_tree(title="Final Tree C", color="orange")
plt.show()

print("Tree C completed.\n")