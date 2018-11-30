from representations import *
import matplotlib.pyplot as plt
import time
import json

SC = StabilityChecker(model_dir='./data/model')
BKWorld = BlockWordEnv(env_file='./xmls/block_world.xml',
                       debug=False,
                       random_color=False,
                       random_num=5)

# shape : half
cuboid_0 = Block([0.1,0.02, 0.02], 'cuboid_0')
cube_0 = Block([0.02,0.02,0.02], 'cube_0')
cube_1 = Block([0.02,0.02,0.02], 'cube_1')
cube_2 = Block([0.02,0.02,0.02], 'cube_2')

table = Table([100,100,0], 'TABLE')


SG_start = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cuboid_0,table, {'position': [0,0]})
SG_start.attach_node(cube_0, cuboid_0, {'position': [0,0]})
SG_start.attach_node(cube_1, cube_0, {'position': [0,0]})
SG_start.attach_node(cube_2, cube_1, {'position': [0,0]})
'''
SG_start.attach_node(block_E, table, {'position': [-40, -40]})
SG_start.attach_node(block_C, block_E, {'position': [0, 0]})
SG_start.attach_node(block_H, block_C, {'position': [0, 0]})
SG_start.attach_node(block_D, block_H, {'position': [0, 0]})
SG_start.attach_node(block_A, block_D, {'position': [0, 0]})
SG_start.attach_node(block_B, block_A, {'position': [0, 0]})

SG_start.attach_node(block_D, table, {'position': [-40, -40]})
SG_start.attach_node(block_C, block_D, {'position': [0, 0]})
SG_start.attach_node(block_B, block_C, {'position': [0, 0]})
SG_start.attach_node(block_A, block_B, {'position': [0, 0]})

SG_start.attach_node(block_C, table, {'position': [20, 0]})
SG_start.attach_node(block_D, table, {'position': [0, 0]})
SG_start.attach_node(block_B, block_D, {'position': [-1.5, 0]})
SG_start.attach_node(block_A, block_D, {'position': [1.5, 0]})
SG_start.attach_node(block_E, block_D, {'position': [0, 0]})
print(SG_start.ConfigList())
'''

SG_goal = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_goal.attach_node(cuboid_0,table, {'position': [0,0]})
SG_goal.attach_node(cube_0, cuboid_0, {'position': [0.08,0]})
SG_goal.attach_node(cube_1, cuboid_0, {'position': [-0.08,0]})
SG_goal.attach_node(cube_2, table, {'position': [0.12,0]})
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
exe_actions = []
for act in actions:
    print('Step', step, ':')
    step = step - 1
    act.show()
    exe_action = act.configMove()
    BKWorld.move_given_blocks(exe_action)
    exe_actions.append(exe_action)
with open('action_seq.json', 'w') as f:
    json.dump(exe_actions, f, indent=2)
#print(actions)
#print(path)
print(len(searchGraph.nodes()))
#nx.draw(searchGraph)
#plt.show()
