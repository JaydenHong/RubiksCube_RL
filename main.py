import cube_env
from cube_env import *
import cube_ursina


surface_colour = {}
surface_colour['U'] = [['Y', 'Y', 'Y'],
                       ['Y', 'Y', 'Y'],
                       ['Y', 'Y', 'Y']]

surface_colour['L'] = [['R', 'R', 'R'],
                       ['B', 'B', 'B'],
                       ['B', 'B', 'B']]

surface_colour['F'] = [['G', 'G', 'G'],
                       ['R', 'R', 'R'],
                       ['R', 'R', 'R']]

surface_colour['R'] = [['O', 'O', 'O'],
                       ['G', 'G', 'G'],
                       ['G', 'G', 'G']]

surface_colour['B'] = [['B', 'B', 'B'],
                       ['O', 'O', 'O'],
                       ['O', 'O', 'O']]

surface_colour['D'] = [['W', 'W', 'W'],
                       ['W', 'W', 'W'],
                       ['W', 'W', 'W']]


def run_cube(seq_str, state):
    seq = seqstr2id(seq_str)
    game = cube_ursina.Game(seq, seq_str, state)
    game.run()


def main():
    c = cube_env.Cube()
    c.printcube()

    seq, seq_str = random_sequence(10, 'HTM')
    seq = seqstr2id('RUR\'U\'')
    c.scramble(seq)
    # c.scramble(seqstr2id('RURURLFBLRBB'))
    # c.encode_color_to_state(surface_colour)

    c.printcube()
    # print(c.surface_colour)
    # run_cube('U', state=None)
    run_cube(seq_str, state=None)

    # pipeline
    # input:
    # a) sequence
    #   random sqe:
    #     seq. seqstr = random_sequence(20, 'HTM')
    #     c.scramble(seq)
    #   given sqe:
    #     seqstr2id('RURURLFBLRBB')
    #     c.scramble(seq)
    # b) color of each facets
    #     c.encode_color_to_state(surface_colour)
    #
    #  -> result: seq, seq_str, state i.e. c.corner_permutation, c.corner_orientation, etc.
    #
    # visualization:
    #    seqstr -> msg
    #    seq -> initialize wt/wo animation
    #
    # RL
    #    - Test
    #    main - c -> ursina -> c.solve : generate seq following policy with len n
    #    in cube.solve, if state is not terminal and initial seq exists, sqn = invseq(seq)
    #    - Train
    #
    # in a,b, corresponding sequence of action id will be generated
    # in c, the order of the center cubies may be different,
    # thus the definition of the right axis and the color mapping must be done first
    # e.g. if the top is red and the left is white, <R->Y, G->R, Y->G>, <W->B, B->O, O->W>
    # also 3d plot must take the order (Top, Left center cubies color info)


if __name__ == '__main__':
    main()
