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
                       random_color=True,
                       random_num=5,
                       random_seed=1619)

cuboid_0 = Block([0.1, 0.02, 0.02], 'cuboid_0')
cuboid_1 = Block([0.1, 0.02, 0.02], 'cuboid_1')
cuboid_2 = Block([0.1, 0.02, 0.02], 'cuboid_2')
cube_0 = Block([0.02, 0.02, 0.02], 'cube_0')
cube_1 = Block([0.02, 0.02, 0.02], 'cube_1')
cube_2 = Block([0.02, 0.02, 0.02], 'cube_2')
cube_3 = Block([0.02, 0.02, 0.02], 'cube_3')
cube_4 = Block([0.02, 0.02, 0.02], 'cube_4')
table = Table([100, 100, 0], 'TABLE')

SG_start = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cuboid_0, table, {'position': [-0.05, 0]})
SG_start.attach_node(cuboid_1, cuboid_0, {'position': [0.02, 0]})
SG_start.attach_node(cuboid_2, cuboid_1, {'position': [-0.04, 0]})
SG_start.attach_node(cube_0, cuboid_2, {'position': [0, 0]})
SG_start.attach_node(cube_1, cube_0, {'position': [0, 0]})
SG_start.attach_node(cube_2, cube_1, {'position': [0, 0]})
SG_start.attach_node(cube_3, cube_2, {'position': [0, 0]})
SG_start.attach_node(cube_4, cube_3, {'position': [0, 0]})

SG_goal = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_goal.attach_node(cube_4, table, {'position': [0.15, 0]})
SG_goal.attach_node(cuboid_0, cube_4, {'position': [0, 0]})
SG_goal.attach_node(cuboid_1, cuboid_0, {'position': [0.02, 0]})
SG_goal.attach_node(cuboid_2, cuboid_1, {'position': [-0.04, 0]})
SG_goal.attach_node(cube_0, cuboid_2, {'position': [-0.01, 0]})
SG_goal.attach_node(cube_1, cube_0, {'position': [0, 0]})
SG_goal.attach_node(cube_2, cuboid_2, {'position': [0.05, 0]})
SG_goal.attach_node(cube_3, cube_2, {'position': [0, 0]})

with open('action_seq.json', 'r') as f:
    exe_actions = json.load(f)

cfgs = SG_start.ConfigList()
BKWorld.reset()
BKWorld.move_given_blocks(cfgs)
BKWorld.step()
start = time.time()
while time.time() - start < 6:
    BKWorld.render()

for exe_action in reversed(exe_actions):
    bk_names = list(exe_action.keys())
    if len(bk_names) > 1:
        BKWorld.move_blocks_for_demo(exe_action)
    else:
        BKWorld.move_block_for_demo(bk_names[0], exe_action[bk_names[0]])
    time.sleep(1)

while True:
    BKWorld.render()
