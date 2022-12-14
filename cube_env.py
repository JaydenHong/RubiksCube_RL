import numpy as np
import pandas as pd
import matplotlib as plt

YELLOW, WHITE, RED, ORANGE, BLUE, GREEN = 0, 1, 2, 3, 4, 5
U, D, F, B, R, L = 0, 1, 2, 3, 4, 5
U_, D_, F_, B_, R_, L_ = 6, 7, 8, 9, 10, 11  # U', D', F', B', R', L'
U2, D2, F2, B2, R2, L2 = 12, 13, 14, 15, 16, 17  # HTM

# Corner Block ID
YBO, YOG, YGR, YRB, WBR, WRG, WGO, WOB = 0, 1, 2, 3, 4, 5, 6, 7

# Edge Block ID
YO, YG, YR, YB, OB, OG, RG, RB, WR, WG, WO, WB = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11


#Mapping
corner_color_map = [['YBO', 'YOG', 'YGR', 'YRB', 'WBR', 'WRG', 'WGO', 'WOB'],  # orientation order 0, corner_id = 0 to 8
                    ['OYB', 'GYO', 'RYG', 'BYR', 'RWB', 'GWR', 'OWG', 'BWO'],  # orientation order 1, corner_id = 0 to 8
                    ['BOY', 'OGY', 'GRY', 'RBY', 'BRW', 'RGW', 'GOW', 'OBW']]  # orientation order 2, corner_id = 0 to 8
edge_color_map = [['YO', 'YG', 'YR', 'YB', 'OB', 'OG', 'RG', 'RB', 'WR', 'WG', 'WO', 'WB'],
                                                                                # orientation order 0, edge_id = 0 to 12
                  ['OY', 'GY', 'RY', 'BY', 'BO', 'GO', 'GR', 'BR', 'RW', 'GW', 'OW', 'BW']]
                                                                                # orientation order 1, edge_id = 0 to 12
# d = dict( (j,(x, y)) for x, i in enumerate(myList) for y, j in enumerate(i) )
# corner_orient_id[orientation_number][permutation_number] = color of the block (eg. 'YBO')
# corner_orient_id[orientation_number][permutation_number][num] = color of the facet (eg. 'Y')

action_list = {'HTM': ['U', 'D', 'F', 'B', 'R', 'L',
                       'U\'', 'D\'', 'F\'', 'B\'', 'R\'', 'L\'',
                       'U2', 'D2', 'F2', 'B2', 'R2', 'L2'],
               'QTM': ['U', 'D', 'F', 'B', 'R', 'L',
                       'U\'', 'D\'', 'F\'', 'B\'', 'R\'', 'L\'']}

state_Terminal = [YBO, YOG, YGR, YRB, WBR, WRG, WGO, WOB]\
                 + [0, 0, 0, 0, 0, 0, 0, 0]\
                 + [YO, YG, YR, YB, OB, OG, RG, RB, WR, WG, WO, WB]\
                 + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# Encode sequence string to the list of sequence IDs
# Input: 'BF2U\'D' / Output:[3, 14, 6, 1]
def seqstr2id(sequence_string):
    sequence = []
    for c in sequence_string:
        try:
            idx = 'UDFBRL2\''.index(c)
            if idx < 6:
                sequence.append(idx)
            elif c == '\'':
                sequence[-1] += 6
            elif c == '2':
                sequence[-1] += 12
        except (Exception,):
            print("invalid sequence")
            return False
    return sequence


# Invert the given sequence
def invseq(sequence):
    sequence.reverse()
    inverse = []
    for action in sequence:
        if action in range(6):
            action += 6
        elif action in range(6,12):
            action -= 6
        inverse.append(action)
    return inverse


# Generate random sequence string and action numbers
# Input: N = length
# Output: 'BF2U\'D', [3, 14, 6, 1]
def random_sequence(n_steps=20, metric='HTM'):
    prev = None
    action = 1
    sequence = []
    for i in range(n_steps):
        while True:  # avoid same two adjacent actions in the sequence such as L2L'
            action = np.random.randint(6)
            if prev != action:
                prev = action
                break
        if metric == 'HTM':
            action = action + np.random.randint(3) * 6  # action id
        elif metric == 'QTM':
            action = action + np.random.randint(2) * 6  # action id

        sequence.append(action)

    seq_letter = action_list[metric]
    sequence_str = ''.join([seq_letter[action] for action in sequence])
    # print(sequence_str)

    return sequence, sequence_str


# Generate episode from random sequence string and ID
# Input: N = length
# Output: [state1, state2, ...], [1, 2, ...]
def generate_episode(n_steps=20, metric='HTM'):
    c = Cube()
    seq, seq_str = random_sequence(n_steps, metric)
    states = c.scramble(seq)
    distance = [i+1 for i in range(n_steps)]
    return states, distance


# Returns all possible next states and rewards
# Input: state
# Output: [state_next_1, ... state_next_18], [-1, -1, +1, -1, ...]
def transitions(state, metric='HTM'):
    action_len = len(action_list[metric])
    c = Cube()
    states_next = []
    rewards = []
    for action in range(action_len):
        # initialize cube with the certain state
        c.corner_permutation = state[0:8]
        c.corner_orientation = state[8:16]
        c.edge_permutation = state[16:28]
        c.edge_orientation = state[28:]
        c.rotate(action)
        s_next = c.corner_permutation + c.corner_orientation + \
                 c.edge_permutation + c.edge_orientation
        states_next.append(s_next)
        if s_next == state_Terminal:
            rewards.append(1)
        else:
            rewards.append(0)
    return states_next, rewards


# seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size = 3
# return: (seq[0:3], seq[3:6], seq[6:9])
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class Cube:
    def __init__(self):
        self.surface_colour = {}
        self.surface_colour['F'] = [['R']*3]*3
        self.surface_colour['B'] = [['O']*3]*3
        self.surface_colour['U'] = [['Y']*3]*3
        self.surface_colour['D'] = [['W']*3]*3
        self.surface_colour['R'] = [['G']*3]*3
        self.surface_colour['L'] = [['B']*3]*3

        # 20 parameters, 8^8 * 12 ^ 12
        self.corner_permutation = [YBO, YOG, YGR, YRB, WBR, WRG, WGO, WOB]  # 0 to 7
        self.corner_orientation = [0, 0, 0, 0, 0, 0, 0, 0]  # 0 to 2
        self.edge_permutation = [YO, YG, YR, YB, OB, OG, RG, RB, WR, WG, WO, WB]    # 0 to 11
        self.edge_orientation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # 0 to 1

        # sum(edge_orientation) mod 2 = 0
        # sum(corner_orientation) mod 3 = 0

        #              c00 e00 c01
        #              e03 [U] e01   # expanded view
        #              c03 e02 c02
        #              c03 e02 c02
        # c00 e03 c03  c03 e02 c02  c02 e01 c01  c01 e00 c00
        # e04 [L] e07  e07 [F] e06  e06 [R] e05  e05 [B] e04
        # c07 e11 c04  c04 e08 c05  c05 e09 c06  c06 e10 c07
        #              c04 e08 c05
        #              e11 [D] e09
        #              c07 e10 c06

        #             c00 e00 c01  # layer view
        #            e03 [U] e01
        #           c03 e02 c02  : 3rd layer

        #             e04 [B] c05
        #            [L] [2] [R]
        #           e07 [F] c06  : 2nd layer

        #             c07 e10 c06
        #            e11 [D] e09
        #           c04 e08 c05  : 1st layer

    def get_state(self):
        return self.corner_permutation + self.corner_orientation + self.edge_permutation + self.edge_orientation

    def printcube(self):

        self.decode_state_to_color()
        for face in 'ULFRBD':
            print([[item for item in row] for row in self.surface_colour[face]])

    def decode_state_to_color(self):

        # get color of each block
        c = ['']*8
        e = ['']*12

        for i in range(8):
            # print(self.corner_orientation[i], self.corner_permutation[i])
            c[i] = corner_color_map[self.corner_orientation[i]][self.corner_permutation[i]]

        for i in range(12):
            e[i] = edge_color_map[self.edge_orientation[i]][self.edge_permutation[i]]


        # map the color to the surface
        self.surface_colour['U'] = [[c[0][0], e[0][0],  c[1][0]],
                                    [e[3][0], 'Y',      e[1][0]],
                                    [c[3][0], e[2][0],  c[2][0]]]
        self.surface_colour['L'] = [[c[0][1], e[3][1],  c[3][2]],
                                    [e[4][1], 'B',      e[7][1]],
                                    [c[7][2], e[11][1], c[4][1]]]
        self.surface_colour['F'] = [[c[3][1], e[2][1],  c[2][2]],
                                    [e[7][0], 'R',      e[6][0]],
                                    [c[4][2], e[8][1],  c[5][1]]]
        self.surface_colour['R'] = [[c[2][1], e[1][1],  c[1][2]],
                                    [e[6][1], 'G',      e[5][1]],
                                    [c[5][2], e[9][1],  c[6][1]]]
        self.surface_colour['B'] = [[c[1][1], e[0][1],  c[0][2]],
                                    [e[5][0], 'O',      e[4][0]],
                                    [c[6][2], e[10][1], c[7][1]]]
        self.surface_colour['D'] = [[c[4][0], e[8][0],  c[5][0]],
                                    [e[11][0], 'W',     e[9][0]],
                                    [c[7][0], e[10][0], c[6][0]]]

    def encode_color_to_state(self, surface_color):
       try:
            # construct color block
            c = ['']*8
            e = ['']*12

            c[0] = surface_color['U'][0][0] + surface_color['L'][0][0] + surface_color['B'][0][2]
            c[1] = surface_color['U'][0][2] + surface_color['B'][0][0] + surface_color['R'][0][2]
            c[2] = surface_color['U'][2][2] + surface_color['R'][0][0] + surface_color['F'][0][2]
            c[3] = surface_color['U'][2][0] + surface_color['F'][0][0] + surface_color['L'][0][2]
            c[4] = surface_color['D'][0][0] + surface_color['L'][2][2] + surface_color['F'][2][0]
            c[5] = surface_color['D'][0][2] + surface_color['F'][2][2] + surface_color['R'][2][0]
            c[6] = surface_color['D'][2][2] + surface_color['R'][2][2] + surface_color['B'][2][0]
            c[7] = surface_color['D'][2][0] + surface_color['B'][2][2] + surface_color['L'][2][0]

            e[0] = surface_color['U'][0][1]+surface_color['B'][0][1]
            e[1] = surface_color['U'][1][2]+surface_color['R'][0][1]
            e[2] = surface_color['U'][2][1]+surface_color['F'][0][1]
            e[3] = surface_color['U'][1][0]+surface_color['L'][0][1]

            e[4] = surface_color['B'][1][2]+surface_color['L'][1][0]
            e[5] = surface_color['B'][1][0]+surface_color['R'][1][2]
            e[6] = surface_color['F'][1][2]+surface_color['R'][1][0]
            e[7] = surface_color['F'][1][0]+surface_color['L'][1][0]

            e[8]  = surface_color['D'][0][1]+surface_color['F'][2][0]
            e[9]  = surface_color['D'][1][2]+surface_color['R'][2][0]
            e[10] = surface_color['D'][2][1]+surface_color['B'][2][0]
            e[11] = surface_color['D'][1][0]+surface_color['L'][2][0]

            print(c,e)

            # find orientation number and permutations

            for i, c in enumerate(c):
                self.corner_orientation[i], self.corner_permutation[i] = \
                 [[orient, perm] for orient, blocklist in enumerate(corner_color_map)
                  for perm, block in enumerate(blocklist) if block == c][0]

            for i, e in enumerate(e):
                self.edge_orientation[i], self.edge_permutation[i] = \
                    [[orient, perm] for orient, blocklist in enumerate(edge_color_map)
                     for perm, block in enumerate(blocklist) if block == e][0]

       except:
           print('impossible color input to construct a cube')

    # Generate a list of states during scramble [s0(=s_T), s1, ... s_n]
    def scramble(self, sequence):
        states = []
        for action in sequence:
            self.rotate(action)
            states.append(self.get_state()) # concatenate cube info into a state

        return states

    # input can be either cube letter such as 'U' or action number such as 0
    def rotate(self, action):
        cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7 = self.corner_permutation
        co0, co1, co2, co3, co4, co5, co6, co7 = self.corner_orientation
        ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11 = self.edge_permutation
        eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11 = self.edge_orientation

        if type(action) == int:
            action_repeat = [1, 3, 2][int(action/6)]
            action = ['U', 'D', 'F', 'B', 'R', 'L'][action % 6]
        elif action in ['U\'', 'D\'', 'F\'', 'B\'', 'R\'', 'L\'']:
            action_repeat = 3
            action = action[0]
        elif action in ['U2', 'D2', 'F2', 'B2', 'R2', 'L2']:
            action_repeat = 2
            action = action[0]
        else:
            return False

        for _ in range(action_repeat):
            # Clockwise turn
            # U: 'YBR' -> 'YRG'
            if action == 'U':
                self.corner_permutation = [cp3, cp0, cp1, cp2, cp4, cp5, cp6, cp7]  # <3210> cyclic
                self.corner_orientation = [co3, co0, co1, co2, co4, co5, co6, co7]  # <3210> cyclic
                self.edge_permutation = [ep3, ep0, ep1, ep2, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3210> cyclic
                self.edge_orientation = [eo3, eo0, eo1, eo2, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11]  # <3210> cyclic
            elif action == 'D':
                self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep11, ep8, ep9, ep10]  # <11,10,9,8> cyclic
                self.corner_permutation = [cp0, cp1, cp2, cp3, cp7, cp4, cp5, cp6]  # <7654> cyclic
                self.edge_orientation = [eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo11, eo8, eo9, eo10]  # <11,10,9,8> cyclic
                self.corner_orientation = [co0, co1, co2, co3, co7, co4, co5, co6]  # <7654> cyclic
            elif action == 'F':
                self.edge_permutation = [ep0, ep1, ep7, ep3, ep4, ep5, ep2, ep8, ep6, ep9, ep10, ep11]  # <2786> cyclic
                self.corner_permutation = [cp0, cp1, cp3, cp4, cp5, cp2, cp6, cp7]  # <2345> cyclic
                self.edge_orientation = [eo0, eo1, (eo7+1) % 2, eo3, eo4, eo5, (eo2+1) % 2, (eo8+1) % 2, (eo6+1) % 2, eo9, eo10, eo11]  # <2786> cyclic
                self.corner_orientation = [co0, co1, (co3+1) % 3, (co4-1) % 3, (co5+1) % 3, (co2-1) % 3, co6, co7]  # <2345> cyclic
            elif action == 'B':
                self.edge_permutation = [ep5, ep1, ep2, ep3, ep0, ep10, ep6, ep7, ep8, ep9, ep4, ep11]  # <0,5,10,4> cyclic
                self.corner_permutation = [cp1, cp6, cp2, cp3, cp4, cp5, cp7, cp0]  # <6701> cyclic
                self.edge_orientation = [(eo5+1) % 2, eo1, eo2, eo3, (eo0+1) % 2, (eo10+1) % 2, eo6, eo7, eo8, eo9, (eo4+1) % 2, eo11]  # <0,5,10,4> cyclic
                self.corner_orientation = [(co1+1) % 3, (co6-1) % 3, co2, co3, co4, co5, (co7+1) % 3, (co0-1) % 3]  # <6701> cyclic
            elif action == 'R':
                self.edge_permutation = [ep0, ep6, ep2, ep3, ep4, ep1, ep9, ep7, ep8, ep5, ep10, ep11]  # <1695> cyclic
                self.corner_permutation = [cp0, cp2, cp5, cp3, cp4, cp6, cp1, cp7]  # <1256> cyclic
                self.edge_orientation = [eo0, eo6, eo2, eo3, eo4, eo1, eo9, eo7, eo8, eo5, eo10, eo11]  # <1695> cyclic
                self.corner_orientation = [co0, (co2+1) % 3, (co5-1) % 3, co3, co4, (co6+1) % 3, (co1-1) % 3, co7]  # <1256> cyclic
            elif action == 'L':
                self.edge_permutation = [ep0, ep1, ep2, ep4, ep11, ep5, ep6, ep3, ep8, ep9, ep10, ep7]  # <3,4,11,7> cyclic
                self.corner_permutation = [cp7, cp1, cp2, cp0, cp3, cp5, cp6, cp4]  # <7430> cyclic
                self.edge_orientation = [eo0, eo1, eo2, eo4, eo11, eo5, eo6, eo3, eo8, eo9, eo10, eo7]  # <3,4,11,7> cyclic
                self.corner_orientation = [(co7-1) % 3, co1, co2, (co0+1) % 3, (co3-1) % 3, co5, co6, (co4+1) % 3]  # <7430> cyclic

        # Counter-clockwise turn
#         elif action == U_:
#             self.edge_permutation = [ep1, ep2, ep3, ep0, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <0123> cyclic
#             self.corner_permutation = [cp1, cp2, cp3, cp0, cp4, cp5, cp6, cp7]  # <0123> cyclic
#             self.edge_orientation = [eo1, eo2, eo3, eo0, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11]
#             self.corner_orientation = [co1, co2, co3, co0, co4, co5, co6, co7]
#
#         elif action == D_:
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <8,9,10,11> cyclic
#             self.corner_permutation = [cp0, cp1, cp2, cp3, cp5, cp6, cp7, cp4]  # <4567> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#
#         elif action == F_:
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <2687> cyclic
#             self.corner_permutation = [cp0, cp1, cp5, cp2, cp3, cp4, cp6, cp7]  # <3254> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#
#         elif action == B_:
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <0,4,5,10> cyclic
#             self.corner_permutation = [cp7, cp0, cp2, cp3, cp4, cp5, cp1, cp6]  # <1076> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#
#         elif action == R_:
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <1596> cyclic
#             self.corner_permutation = [cp0, cp6, cp1, cp3, cp4, cp2, cp5, cp7]  # <2165> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#
#         elif action == L_:
#             self.corner_permutation = [cp3, cp1, cp2, cp4, cp7, cp5, cp6, cp0]  # <0347> cyclic
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3,7,11,4> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#
# # HTM
#         elif action == U2:
#             self.corner_permutation = [cp2, cp3, cp0, cp1, cp4, cp5, cp6, cp7]  # <31><20> cyclic
#             self.corner_orientation = [co2, co3, co0, co1, co4, co5, co6, co7]
#             self.edge_permutation = [ep3, ep0, ep1, ep2, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3210> cyclic
#             self.edge_orientation = [eo3, eo0, eo1, eo2, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11]  # <3210> cyclic
#
#         elif action == D2:
#             self.corner_permutation = [cp0, cp1, cp2, cp3, cp6, cp7, cp4, cp5]  # <75><64> cyclic
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3,7,11,4> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#
#         elif action == F2:
#             self.corner_permutation = [cp0, cp1, cp4, cp5, cp2, cp3, cp6, cp7]  # <42><53> cyclic
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3,7,11,4> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#
#         elif action == B2:
#             self.corner_permutation = [cp6, cp7, cp2, cp3, cp4, cp5, cp0, cp1]  # <60><71> cyclic
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3,7,11,4> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#
#         elif action == R2:
#             self.corner_permutation = [cp0, cp5, cp6, cp3, cp4, cp1, cp2, cp7]  # <51><62> cyclic
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3,7,11,4> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]
#
#         elif action == L2:
#             self.corner_permutation = [cp4, cp1, cp2, cp7, cp0, cp5, cp6, cp3]  # <73><40> cyclic
#             self.corner_orientation = [cp0, cp1, (cp3 + 1) % 3, (cp4 - 1) % 3, (cp5 + 1) % 3, (cp2 - 1) % 3, cp6, cp7]
#             self.edge_permutation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]  # <3,7,11,4> cyclic
#             self.edge_orientation = [ep0, ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, ep11]

        # print(self.corner_permutation, '\n', self.corner_orientation, '\n', self.edge_permutation, '\n', self.edge_orientation)