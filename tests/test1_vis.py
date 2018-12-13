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
                       random_seed=12)

# shape : half
cuboid_0 = Block([0.1, 0.02, 0.02], 'cuboid_0')
cube_0 = Block([0.02, 0.02, 0.02], 'cube_0')
cube_1 = Block([0.02, 0.02, 0.02], 'cube_1')
cube_2 = Block([0.02, 0.02, 0.02], 'cube_2')

table = Table([100, 100, 0], 'TABLE')

SG_start = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_start.attach_node(cube_0, table, {'position': [-0.06, 0]})
SG_start.attach_node(cuboid_0, cube_0, {'position': [0, 0]})
SG_start.attach_node(cube_1, cuboid_0, {'position': [0.05, 0]})
SG_start.attach_node(cube_2, cuboid_0, {'position': [-0.05, 0]})

SG_goal = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_goal.attach_node(cuboid_0, table, {'position': [0.06, 0]})
SG_goal.attach_node(cube_0, cuboid_0, {'position': [0, 0]})
SG_goal.attach_node(cube_1, cuboid_0, {'position': [0.05, 0]})
SG_goal.attach_node(cube_2, cuboid_0, {'position': [-0.05, 0]})

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
