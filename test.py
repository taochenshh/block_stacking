from representations import *
import matplotlib.pyplot as plt
import time

# shape : half
block_D = Block([5,2,1], 'BLOCK_D')
block_A = Block([2,2,2], 'BLOCK_A')
block_B = Block([2,2,2], 'BLOCK_B')
block_C = Block([2,2,2], 'BLOCK_C')
table = Table([100,100,0], 'TABLE')
block_E = Block([2,2,2], 'BLOCK_E')
block_H = Block([2,2,5], 'BLOCK_H')

SG_start = StateGraph(table)
'''
SG_start.attach_node(block_E, table, {'position': [-40, -40]})
SG_start.attach_node(block_C, block_E, {'position': [0, 0]})
SG_start.attach_node(block_H, block_C, {'position': [0, 0]})
SG_start.attach_node(block_D, block_H, {'position': [0, 0]})
SG_start.attach_node(block_A, block_D, {'position': [0, 0]})
SG_start.attach_node(block_B, block_A, {'position': [0, 0]})
'''
'''
SG_start.attach_node(block_D, table, {'position': [-40, -40]})
SG_start.attach_node(block_C, block_D, {'position': [0, 0]})
SG_start.attach_node(block_B, block_C, {'position': [0, 0]})
SG_start.attach_node(block_A, block_B, {'position': [0, 0]})
'''

SG_start.attach_node(block_C, table, {'position': [20, 0]})
SG_start.attach_node(block_D, table, {'position': [0, 0]})
SG_start.attach_node(block_B, block_D, {'position': [-1.5, 0]})
SG_start.attach_node(block_A, block_D, {'position': [1.5, 0]})
SG_start.attach_node(block_E, block_D, {'position': [0, 0]})
print(SG_start.ConfigList())

SG_goal = StateGraph(table)
'''
SG_goal.attach_node(block_D, table, {'position': [20, 20]})
SG_goal.attach_node(block_C, block_D, {'position': [0, 0]})
SG_goal.attach_node(block_E, block_C, {'position': [0, 0]})
SG_goal.attach_node(block_B, block_C, {'position': [-1.5, 0]})
SG_goal.attach_node(block_A, block_C, {'position': [1.5, 0]})
SG_goal.attach_node(block_H, block_E, {'position': [0, 0]})
'''

'''
SG_goal.attach_node(block_C, table, {'position': [-40, -40]})
SG_goal.attach_node(block_D, block_C, {'position': [0, 0]})
SG_goal.attach_node(block_A, block_D, {'position': [0, 0]})
#SG_goal.attach_node(block_B, block_D, {'position': [0, 0]})
'''
'''
SG_goal.attach_node(block_A, table, {'position': [-40, -40]})
SG_goal.attach_node(block_C, block_A, {'position': [0, 0]})
SG_goal.attach_node(block_B, block_C, {'position': [0, 0]})
SG_goal.attach_node(block_D, block_B, {'position': [0, 0]})
'''
'''
SG_goal.attach_node(block_C, table, {'position': [-40, -40]})
SG_goal.attach_node(block_B, block_C, {'position': [0, 0]})
SG_goal.attach_node(block_A, block_B, {'position': [0, 0]})
SG_goal.attach_node(block_D, block_A, {'position': [0, 0]})
'''

SG_goal.attach_node(block_A, table, {'position': [20, 0]})
SG_goal.attach_node(block_D, block_A, {'position': [0, 0]})
SG_goal.attach_node(block_B, block_D, {'position': [-1.5, 0]})
SG_goal.attach_node(block_C, block_D, {'position': [1.5, 0]})

'''
EIS = ExactIdenticalSubgraph(SG_start, SG_goal)
CIS = ConfigIdenticalSubgraph(SG_start, SG_goal)
#SG_goal.dettach_node(block_C)

plt.subplot(121)
nx.draw(SG_start.graph, with_labels=True, font_weight='regular')
plt.axis('off')
plt.subplot(122)
nx.draw(SG_goal.graph, with_labels=True, font_weight='regular')
plt.axis('off')
plt.show()
'''
#print(computeHeuristics(SG_start, SG_goal))
'''
acts = expandActions(SG_start, SG_goal)
for act in acts:
    act.show()
'''
'''
SG_copy = SG_start.copy()
SG_copy.dettach_node(block_A)
print(SG_start.layers)
print(SG_copy.layers)
'''
start_time = time.time()
actions, path, searchGraph = DFSearch(SG_start, SG_goal)
print('Planning Time:', time.time() - start_time, 'sec')
step = len(actions)
for act in actions:
    print('Step', step, ':')
    step = step - 1
    act.show()
    print(act.configMove())
#print(actions)
#print(path)
print(len(searchGraph.nodes()))
#nx.draw(searchGraph)
#plt.show()
