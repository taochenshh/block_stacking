from representations import *
import matplotlib.pyplot as plt
import time
import json

SC = StabilityChecker(model_dir='./data/model')
BKWorld = BlockWordEnv(env_file='./xmls/block_world.xml',
                       debug=False,
                       random_color=True,
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

SG_goal = StateGraph(table, SC=SC, BKWorld=BKWorld)

SG_goal.attach_node(cuboid_0, table, {'position': [0,0]})
SG_goal.attach_node(cube_0, cuboid_0, {'position': [0.08,0]})
SG_goal.attach_node(cube_1, cuboid_0, {'position': [-0.08,0]})
SG_goal.attach_node(cube_2, table, {'position': [0.12,0]})


with open('action_seq.json', 'r') as f:
    exe_actions = json.load(f)

cfgs = SG_start.ConfigList()
BKWorld.reset()
BKWorld.move_given_blocks(cfgs)
BKWorld.step()
BKWorld.render()
time.sleep(2)

for exe_action in reversed(exe_actions):
    for bk_name, pos in exe_action.items():
        BKWorld.move_block_for_demo(bk_name, pos)

while True:
    BKWorld.render()


