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
cube_3 = Block([0.02,0.02,0.02], 'cube_3')

table = Table([100,100,0], 'TABLE')


SG_start = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cube_0,table, {'position': [-0.06,0]})
SG_start.attach_node(cuboid_0, cube_0, {'position': [0,0]})
SG_start.attach_node(cube_1, cuboid_0, {'position': [0.05,0]})
SG_start.attach_node(cube_2, cuboid_0, {'position': [-0.05,0]})

SG_goal = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_goal.attach_node(cuboid_0,table, {'position': [0.06,0]})
SG_goal.attach_node(cube_0, cuboid_0, {'position': [0,0]})
SG_goal.attach_node(cube_1, cuboid_0, {'position': [0.05,0]})
SG_goal.attach_node(cube_2, cuboid_0, {'position': [-0.05,0]})

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
    print(exe_action)
    BKWorld.move_given_blocks(exe_action)
    exe_actions.append(exe_action)
with open('action_seq.json', 'w') as f:
    json.dump(exe_actions, f, indent=2)

print('nodes in searchGraph:',len(searchGraph.nodes()))
