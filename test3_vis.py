from representations import *
import time
import json

SC = StabilityChecker(model_dir='./data/model')
BKWorld = BlockWordEnv(env_file='./xmls/block_world.xml',
                       debug=False,
                       random_color=False,
                       random_num=5)
# shape : half
cuboid_2 = Block([0.1, 0.02, 0.02], 'cuboid_2')
cuboid_5 = Block([0.1, 0.02, 0.02], 'cuboid_5')
cuboid_7 = Block([0.1, 0.02, 0.02], 'cuboid_7')

cube_1 = Block([0.02, 0.02, 0.02], 'cube_1')
cube_3 = Block([0.02, 0.02, 0.02], 'cube_3')
cube_4 = Block([0.02, 0.02, 0.02], 'cube_4')
cube_6 = Block([0.02, 0.02, 0.02], 'cube_6')
cube_8 = Block([0.02, 0.02, 0.02], 'cube_8')
cube_9 = Block([0.02, 0.02, 0.02], 'cube_9')
table = Table([100, 100, 0], 'TABLE')

SG_start = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cube_1, table, {'position': [-0.08, 0]})
SG_start.attach_node(cuboid_2, cube_1, {'position': [0.0, 0]})
SG_start.attach_node(cube_3, cuboid_2, {'position': [-0.08, 0]})
SG_start.attach_node(cube_4, cuboid_2, {'position': [0.08, 0]})
SG_start.attach_node(cuboid_5, cube_3, {'position': [0.08, 0]})
SG_start.attach_node(cuboid_5, cube_4, {'position': [-0.08, 0]})
SG_start.attach_node(cube_6, cuboid_5, {'position': [0.0, 0]})
SG_start.attach_node(cuboid_7, cube_6, {'position': [-0.0, 0]})
SG_start.attach_node(cube_8, cuboid_7, {'position': [-0.08, 0]})
SG_start.attach_node(cube_9, cuboid_7, {'position': [0.08, 0]})
print(SG_start.ifStable())

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

SG_goal.attach_node(cuboid_5, table, {'position': [0.14, 0]})
SG_goal.attach_node(cube_6, cuboid_5, {'position': [0.0, 0]})
SG_goal.attach_node(cube_8, cube_6, {'position': [0, 0]})
SG_goal.attach_node(cube_9, cube_8, {'position': [0, 0]})
SG_goal.attach_node(cuboid_7, cube_9, {'position': [0.0, 0]})
SG_goal.attach_node(cuboid_2, cuboid_7, {'position': [0.0, 0]})
SG_goal.attach_node(cube_3, cuboid_2, {'position': [-0.08, 0]})
SG_goal.attach_node(cube_4, cuboid_2, {'position': [0.08, 0]})
SG_goal.attach_node(cube_1, cuboid_2, {'position': [0, 0]})
print(SG_goal.ifStable())
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
# print(computeHeuristics(SG_start, SG_goal))
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

with open('action_seq.json', 'r') as f:
    exe_actions = json.load(f)

cfgs = SG_start.ConfigList()
BKWorld.reset()
BKWorld.move_given_blocks(cfgs)
BKWorld.step()
BKWorld.render()
start = time.time()
while time.time() - start < 5:
    BKWorld.render()

for exe_action in reversed(exe_actions):
    bk_names = list(exe_action.keys())
    if len(bk_names) > 1:
        BKWorld.move_blocks_for_demo(exe_action)
    else:
        BKWorld.move_block_for_demo(bk_names[0], exe_action[bk_names[0]])

while True:
    BKWorld.render()