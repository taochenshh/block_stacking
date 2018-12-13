import json
import time
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(cur_dir))
from symbolic_planning import *

SC = StabilityChecker(model_dir='../data/model')
BKWorld = BlockWordEnv(env_file='../xmls/block_world.xml',
                       debug=False,
                       random_color=False,
                       random_num=5)

# shape : half
cuboid_5 = Block([0.1, 0.02, 0.02], 'cuboid_0')
cuboid_7 = Block([0.1, 0.02, 0.02], 'cuboid_1')

cube_1 = Block([0.02, 0.02, 0.02], 'cube_1')
cube_2 = Block([0.02, 0.02, 0.02], 'cube_2')
cube_3 = Block([0.02, 0.02, 0.02], 'cube_3')
cube_4 = Block([0.02, 0.02, 0.02], 'cube_4')
cube_6 = Block([0.02, 0.02, 0.02], 'cube_5')
cube_8 = Block([0.02, 0.02, 0.02], 'cube_6')
cube_9 = Block([0.02, 0.02, 0.02], 'cube_0')
table = Table([100, 100, 0], 'TABLE')

SG_start = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cube_1, table, {'position': [-0.04, 0]})
SG_start.attach_node(cube_3, table, {'position': [0.04, 0]})
SG_start.attach_node(cube_2, cube_1, {'position': [0, 0]})
SG_start.attach_node(cube_4, cube_3, {'position': [0, 0]})
SG_start.attach_node(cuboid_5, cube_2, {'position': [0.04, 0]})
SG_start.attach_node(cube_6, cuboid_5, {'position': [0.0, 0]})
SG_start.attach_node(cuboid_7, cube_6, {'position': [-0.08, 0]})
SG_start.attach_node(cube_8, cuboid_7, {'position': [0.08, 0]})
SG_start.attach_node(cube_9, cube_8, {'position': [0, 0]})

SG_goal = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cuboid_5, table, {'position': [0.2, 0]})
SG_start.attach_node(cube_6, cuboid_5, {'position': [0.0, 0]})
SG_start.attach_node(cube_8, cube_6, {'position': [0, 0]})
SG_start.attach_node(cube_9, cube_8, {'position': [0, 0]})
SG_start.attach_node(cuboid_7, cube_9, {'position': [0.0, 0]})
SG_start.attach_node(cube_1, cuboid_7, {'position': [0.0, 0]})
SG_start.attach_node(cube_2, cube_1, {'position': [0, 0]})
SG_start.attach_node(cube_3, cube_2, {'position': [0.0, 0]})
SG_start.attach_node(cube_4, cube_3, {'position': [0, 0]})

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
print(len(searchGraph.nodes()))

