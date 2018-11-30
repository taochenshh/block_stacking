import networkx as nx
import copy
import numpy as np
from stability_checker import StabilityChecker
from env.block_world import BlockWordEnv
import cv2

class StateGraph:
    def __init__(self, root_node, SC, BKWorld):
        # root_nodes: root node at layer 0
        self.graph = nx.DiGraph()
        self.root_node = root_node
        self.graph.add_node(root_node)
        self.layers = list([])
        self.layers.append([root_node])
        self.graph.nodes[root_node]['layer'] = 0
        self.graph.nodes[root_node]['abs_position'] = [0,0,0]
        self.SC = SC
        self.BKWorld = BKWorld
        #i = [x for x in range(len(c)) if 1 in c[x]]

    def manipulable_nodes(self):
        # return the upmost objects
        return [x for x in list(self.graph.nodes()) if self.graph.out_degree(x) == 0]

    def attach_node(self, node, pre_node, edge_info):
        # edge_info: dictionary of attributes of edge
        self.graph.add_edges_from([(pre_node, node, edge_info)])
        cur_layer = self.graph.nodes[pre_node]['layer'] + 1
        self.graph.nodes[node]['layer'] = cur_layer
        p_change = edge_info['position'] + [pre_node.shape[2]+node.shape[2]]
        self.graph.nodes[node]['abs_position'] = list(np.array(self.graph.nodes[pre_node]['abs_position']) + np.array(p_change))
        if len(self.layers) < cur_layer + 1:
            self.layers = self.layers + [[node]]
        else:
            self.layers[cur_layer] = self.layers[cur_layer] + [node]

    def dettach_node(self, node):
        n_index = self.graph.nodes[node]['layer']
        self.graph.nodes[node]['layer'] = -1
        self.layers[n_index].remove(node)
        self.graph.remove_node(node)

    def ConfigList(self):
        Clist = {x.name: self.graph.nodes[x]['abs_position'] for x in list(self.graph.nodes())
                 if self.graph.nodes[x]['layer'] != 0}
        return Clist

    def ifStable(self):
        blockConfigurations = self.ConfigList()
        self.BKWorld.reset()
        self.BKWorld.move_given_blocks(blockConfigurations)
        self.BKWorld.step()
        img = self.BKWorld.get_img()
        cv2.imwrite('test.png', img)
        label = self.BKWorld.check_stability(render=False)
        return label

    def ifCollide(self):
        return False

    def copy(self):
        CSG = StateGraph(self.root_node, SC=self.SC, BKWorld=self.BKWorld)
        CSG.graph = nx.DiGraph(self.graph)
        CSG.layers = [x[:] for x in self.layers]
        return CSG

    def ifSame(self, SG):
        EISub = ExactIdenticalSubgraph(self, SG)
        if EISub.number_of_nodes() == SG.graph.number_of_nodes() and EISub.number_of_nodes() == self.graph.number_of_nodes():
            return True
        else:
            return False

    def recomputeLayers(self):
        self.layers = []
        last_layer = [x for x in list(self.graph.nodes()) if self.graph.in_degree(x) == 0]
        self.root_node = last_layer[0]
        self.layers.append(last_layer)
        layer_index = 0
        while not len(last_layer) == 0:
            this_layer = []
            for node in last_layer:
                this_layer = this_layer + list(self.graph.successors(node))
                self.graph.nodes[node]['layer'] = layer_index
            if not len(this_layer) == 0:
                self.layers.append(this_layer)
            last_layer = this_layer
            layer_index = layer_index + 1

    def subgraph(self, nodes):
        SSG = StateGraph(nodes[0])
        SSG.graph = nx.DiGraph(nx.subgraph(self.graph, nodes))
        SSG.recomputeLayers()
        return SSG

'''
class NodeObject:
    def __init__(self):

class EdgeRelation:
'''

class Block:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

class Table:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

class Action:
    def __init__(self, action_options, SG):
        self.opt = action_options
        self.state_graph = SG

    def ifeligible(self):
        return self.check_precondition() and self.check_actcondition() and self.check_poscondition()

    def check_precondition(self):
        return True

    def check_actcondition(self):
        return True

    def check_poscondition(self):
        return True

class MoveTo(Action):
    # action options: (ObjectToMove, ObjectToPutOn, postion(x,y,z,quaternions))
    def check_precondition(self):
        MoveNode = self.opt[0]
        if self.state_graph.graph.out_degree(MoveNode) == 0:
            return True
        else:
            return False

    def check_actcondition(self):
        act_SG = self.state_graph.copy()
        act_SG.dettach_node(self.opt[0])
        return act_SG.ifStable()

    def check_poscondition(self):
        self.add_effect()
        return self.pos_state_graph.ifStable() and not self.pos_state_graph.ifCollide()

    def add_effect(self):
        pos_SG = self.state_graph.copy()
        pos_SG.dettach_node(self.opt[0])
        pos_SG.attach_node(self.opt[0], self.opt[1], {'position': self.opt[2]})
        self.pos_state_graph = pos_SG

    def show(self):
        print('Move block:', self.opt[0].name, 'to', self.opt[1].name, 'at position', self.opt[2])

    def configMove(self):
        return {self.opt[0].name: self.pos_state_graph.graph.nodes[self.opt[0]]['abs_position']}

class MoveSubTo(Action):
    # action options: (SubassemblyToMove, ObjectToPutOn, root_node postion(x,y,z,quaternions))
    # SubassemblyToMove: list of nodes, list[0] is the root node of this Subassembly
    def check_actcondition(self):
        act_SG = self.state_graph.copy()
        for node in self.opt[0]:
            act_SG.dettach_node(node)
        return act_SG.ifStable()

    def check_poscondition(self):
        self.add_effect()
        return self.pos_state_graph.ifStable() and not self.pos_state_graph.ifCollide()

    def add_effect(self):
        pos_SG = self.state_graph.copy()
        for node in list(self.state_graph.graph.predecessors(self.opt[0][0])):
            pos_SG.graph.remove_edge(node, self.opt[0][0])
        pos_SG.graph.add_edges_from([(self.opt[1], self.opt[0][0], {'position': self.opt[2]})])
        pos_SG.recomputeLayers()
        self.pos_state_graph = pos_SG

    def show(self):
        nodes_name = '['
        for node in self.opt[0]:
            nodes_name = nodes_name + node.name + ','
        nodes_name = nodes_name + ']'
        print('Move Subassembly:', nodes_name, 'to', self.opt[1].name, 'at position', self.opt[2])

    def configMove(self):
        return {node.name: self.pos_state_graph.graph.nodes[node]['abs_position'] for node in self.opt[0]}


def ExactIdenticalSubgraph(SG1, SG2):
    # return a none layered subgraph
    all_ide_nodes = []
    EIS = nx.DiGraph()
    for d_layer in range(min([len(SG1.layers),len(SG2.layers)])):
        layer_ide_nodes = []
        common_nodes = [x for x in SG1.layers[d_layer] if x in SG2.layers[d_layer]]
        for node in common_nodes:
            ifedges = False
            if list(SG1.graph.predecessors(node)) == list(SG2.graph.predecessors(node)):
                ifedges = True
                for p in SG1.graph.predecessors(node):
                    if not SG1.graph.edges[p,node] == SG2.graph.edges[p,node]:
                        ifedges = False

            if ifedges:
                layer_ide_nodes = layer_ide_nodes + [node]

        all_ide_nodes = all_ide_nodes + layer_ide_nodes
        if len(layer_ide_nodes) == 0:
            break

    return nx.DiGraph(nx.subgraph(SG1.graph,all_ide_nodes))

def ConfigIdenticalSubgraph(SG1, SG2):
    EISub = ExactIdenticalSubgraph(SG1, SG2)
    #print(list(EISub.nodes()))
    # remove IdeSUb from SG1 and SG2
    G1 = nx.DiGraph(SG1.graph)
    G2 = nx.DiGraph(SG2.graph)
    G1.remove_nodes_from(list(EISub.nodes()))
    G2.remove_nodes_from(list(EISub.nodes()))
    CISub = nx.DiGraph(EISub)
    #print(list(EISub.nodes()))
    #
    common_nodes = [x for x in list(G1.nodes()) if x in list(G2.nodes())]
    for node in common_nodes:
        common_successors = [x for x in list(G1.successors(node)) if x in list(G2.successors(node))]
        for successor in common_successors:
            if G1.edges[node, successor] == G2.edges[node, successor]:
                CISub.add_edges_from([(node, successor, G1.edges[node, successor])])
                #CISub.edges[node, successor] = G1.edges[node, successor]

    return CISub

def computeHeuristics(SG_cur, SG_goal):

    '''
    IdeSub = IdenticalSubgraph(SG_cur, SG_goal)
    Nodes_ide = list(IdeSub.nodes)
    Nodes_cur = list(SG_cur.graph.nodes)
    Nodes_goal = list(SG_goal.graph.nodes)
    Nodes_com = [x for x in Nodes_cur if x in Nodes_goal and not x in Nodes_ide]

    return len(Nodes_goal) + len(Nodes_cur) - 2*len(Nodes_ide) - len(Nodes_com )
    '''
    CISub = ConfigIdenticalSubgraph(SG_cur, SG_goal)

    Nodes_ide = list(CISub.nodes())
    Nodes_ide.remove(SG_cur.layers[0][0])
    Nodes_cur = list(SG_cur.graph.nodes())
    Nodes_cur.remove(SG_cur.layers[0][0])
    Nodes_goal = list(SG_goal.graph.nodes())
    Nodes_goal.remove(SG_goal.layers[0][0])
    Nodes_com = [x for x in Nodes_cur if x in Nodes_goal and not x in Nodes_ide]

    Nodes_extra = [x for x in Nodes_cur if not x in Nodes_goal]
    Nodes_irrelevant = []
    for node in Nodes_extra:
        relevant_adj = [x for x in list(SG_cur.graph.adj[node]) if x in Nodes_goal]
        if len(relevant_adj) == 0:
            Nodes_irrelevant = Nodes_irrelevant + [node]

    return len(Nodes_goal) + len(Nodes_cur) - 2*len(Nodes_ide) - len(Nodes_com) - len(Nodes_irrelevant)

def DFSearch(SG_start, SG_goal):
    searchGraph = nx.DiGraph()

    searchGraph.add_node(SG_start)
    searchGraph.nodes[SG_start]['Expanded'] = False
    searchGraph.nodes[SG_start]['heuristics'] = computeHeuristics(SG_start, SG_goal)
    cur_node = SG_start
    NoSolutionFound = False
    GoalFound = False
    while not NoSolutionFound and not GoalFound:#stop condition is not satisfied:

        # compare current node and goal
        EISub = ExactIdenticalSubgraph(cur_node, SG_goal)
        if EISub.number_of_nodes() == SG_goal.graph.number_of_nodes():

            GoalFound = True
            break

        # if current node is not expanded (actions are not generated)
        if searchGraph.nodes[cur_node]['Expanded'] == False:
            cur_actions = expandActions(cur_node, SG_goal)
            searchGraph.nodes[cur_node]['Expanded'] = True

            if len(cur_actions) == 0: # if node cannot be expanded,
                if cur_node == SG_start:
                    NoSolutionFound = True
                    print('no more actions')
                    break
                else:
                    cur_node = list(searchGraph.predecessors(cur_node))[0]
            else: # expand nodes
                for act in cur_actions:
                    # for each action, find out nodes
                    SG_posact = act.pos_state_graph
                    ifnodeExist = False
                    for cmpnode in list(searchGraph.nodes()):
                        ifnodeExist = ifnodeExist or SG_posact.ifSame(cmpnode)
                    if ifnodeExist:
                        continue
                    # compute node Heuristics
                    heu = computeHeuristics(SG_posact, SG_goal)
                    '''
                    if isinstance(act, MoveSubTo):
                        print('sub heu:', heu)
                    '''
                    # add edge to cur_node and posact_node
                    searchGraph.add_edge(cur_node, SG_posact, action = act)
                    searchGraph.nodes[SG_posact]['heuristics'] = heu
                    searchGraph.nodes[SG_posact]['Expanded'] = False
                # find next node
                cur_successors = list(searchGraph.successors(cur_node))
                if len(cur_successors) == 0:
                    if cur_node == SG_start:
                        NoSolutionFound = True
                        print('no valid actions')
                        break
                    else:
                        cur_node = list(searchGraph.predecessors(cur_node))[0]
                        continue
                # compare f = g+h
                heus = [searchGraph.nodes[x]['heuristics'] for x in cur_successors]
                # choose a cur_node
                cur_node = cur_successors[heus.index(min(heus))]

        else: # node expanded, find its unexpanded successors of lowest f = g + h
            cur_successors = [x for x in list(searchGraph.successors(cur_node)) if not searchGraph.nodes[x]["Expanded"]]

            if len(cur_successors) == 0:
                if cur_node == SG_start:
                    NoSolutionFound = True
                    print('no more nodes to explore')
                    break
                else:
                    cur_node = list(searchGraph.predecessors(cur_node))[0]
                    continue

            # compare f = g+h
            heus = [searchGraph.nodes[x]['heuristics'] for x in cur_successors]
            # choose a cur_node
            cur_node = cur_successors[heus.index(min(heus))]

    actions = []
    path = []
    print('GoalFound:', GoalFound)
#    print('No Solution Found:', NoSolutionFound)
    if GoalFound:
        while not cur_node == SG_start:
            path.append(cur_node)
            actions.append(searchGraph.edges[list(searchGraph.in_edges(cur_node))[0]]['action'])
            cur_node = list(searchGraph.predecessors(cur_node))[0]

    return actions, path, searchGraph

def expandActions(SG, SG_goal):
    actions = []
    CISub = ConfigIdenticalSubgraph(SG, SG_goal)
    # build actions
    movable_nodes = [x for x in SG.manipulable_nodes() if x not in list(CISub.nodes())]
    for node in list(CISub.nodes()):
        possible_nodes = [x for x in list(SG_goal.graph.successors(node)) if x in movable_nodes]
        if len(possible_nodes) == 0:
            continue
        for pnode in possible_nodes:
            cur_act = MoveTo([pnode, node, SG_goal.graph.edges[node, pnode]['position']], SG)
            if cur_act.ifeligible():
                actions.append(cur_act)

    # set block on-table action
    for node in SG.manipulable_nodes():
        position = [] # TODO randomly find a space on the table
        cur_act = MoveTo([node, SG.root_node, position], SG)
        if cur_act.ifeligible():
            actions.append(cur_act)

    # try sub assembly

    # find single root node stable subgraph & the root node is the successor of current exact same graph
    EISub = ExactIdenticalSubgraph(SG, SG_goal)
    CISub.remove_nodes_from(EISub)
    roots = [x for x in list(CISub.nodes()) if CISub.in_degree(x) == 0]
    for rnode in roots:
        pre_node = [x for x in list(EISub.nodes()) if rnode in list(SG_goal.graph.successors(x))]
        if len(pre_node) == 0:
            continue
        pre_node = pre_node[0]
        sub_closed = []
        sub_open = [rnode]
        while not len(sub_open) == 0:
            for node in sub_open:
                sub_closed.append(node)
                sub_open.remove(node)
                sub_open = sub_open + list(CISub.successors(node))
        if len(sub_closed) < 2:
            continue
        subG = SG.subgraph(sub_closed)
        if subG.ifStable():
            cur_act = MoveSubTo([sub_closed, pre_node, SG_goal.graph.edges[pre_node, rnode]['position']], SG)
            if cur_act.ifeligible():
                actions.append(cur_act)

    return actions
