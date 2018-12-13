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
