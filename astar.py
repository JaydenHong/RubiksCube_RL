
import cube_env
from cube_env import *
from Train_MC import get_model
import matplotlib.pyplot as plt
from tqdm import tqdm
# N_SCRAMBLE = 10  # the number of scrambles from the initial cube


class Node:
    def __init__(self, parent=None, action='', state=None):
        self.parent = parent
        self.action = action
        self.state = state
        self.g = 0
        self.h = 0  # h = v
        self.f = 0  # f = h+g

    def __eq__(self, other):
        return self.state == other.state


def astar(seq, model):
    # Gen Rand state
    c = cube_env.Cube()
    c.scramble(seq)
    state_initial = c.get_state()

    # Create start and end node
    start_node = Node(state=state_initial)
    end_node = Node(state=state_Terminal)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    for _ in tqdm(range(1000)):
        if not len(open_list) > 0:
            break
        # print('iteration', i)
        # for node in open_list:
        #     print(node.state)
        #     print(node.action,node.g, node.h,'\n')
        # Get the current node with minimum cost
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index  # index in the open list to pop out
        # print('current_node:')
        # print(current_node.state)
        # print(current_node.action, current_node.g, current_node.h, '\n')
        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the path when it reached to the terminal
        if current_node == end_node:
            path = []
            node_on_path = current_node
            while node_on_path is not None:
                path.append(node_on_path.action)
                node_on_path = node_on_path.parent
            return True, ''.join([str(action) for action in path[::-1]])  # return path

        # Generate next states as children
        children = []
        for action, state_next in enumerate(transitions(current_node.state)[0]):
            new_node = Node(parent=current_node, action=action_list['HTM'][action], state=state_next)
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is in the closed list
            conti_flag = False
            for closed_child in closed_list:
                if child == closed_child:
                    conti_flag = True
                    break
            if conti_flag:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = model.predict(np.array([child.state])).ravel()[0] * -1
            child.f = child.g + child.h

            # Child is already in the open list
            for idx, open_node in enumerate(open_list):
                if child == open_node:
                    if child.g < open_node.g:
                        open_list[idx] = child   # update duplicate node in open list to new finding
                    conti_flag = True
                    break
            if conti_flag:
                continue

            # Add the child to the open list
            open_list.append(child)

    path = []
    node_on_path = current_node
    while node_on_path is not None:
        path.append(node_on_path.action)
        node_on_path = node_on_path.parent
    return False, ''.join([str(action) for action in path[::-1]])  # Return reversed path


def greedy(seq, model):
    # Gen Rand state
    c = cube_env.Cube()
    c.scramble(seq)
    solution = ''
    for i in range(len(seq)+50):
        s_next_all = transitions(c.get_state())[0]
        value_next = model.predict(np.array(s_next_all)).ravel()
        # print(value_next)
        actions = value_next.argsort()
        a = action_list['HTM'][actions[-1]]
        # for k in range(18):
        #     a = action_list['HTM'][actions[-1-k]]
        #     if not a[0] == prev_a[0]:
        #         break
        # a = action_list['HTM'][value_next.argmax()]
        solution += a
        c.rotate(a)
        prev_a = a
        # print(c.get_state())
        if c.get_state() == state_Terminal:
            return True, solution
    return False, solution

def evaluate(algorithm, scramble_range, N_TEST=100):
    file_path = 'weights_MC.h5'
    model = get_model()
    model.load_weights(file_path)
    # Read weights
    solved = [0]*20
    for n in range(scramble_range[0]-1, scramble_range[1]):
        for _ in range(N_TEST):
            # Gen Rand state
            seq, seq_str = random_sequence(n+1, metric='HTM')
            success, solution = algorithm(seq, model)
            print('')
            print(seq_str, ': problem, length = ', len(seq))
            print(seqid2str(invseq(seq)), ': answer(Inverse)')
            if success:
                print(solution, ': solved')
                solved[n] += 1
            else:
                print(solution, ': failed <----')
    print(solved)
    plt.bar(range(scramble_range[0], scramble_range[1]+1), solved[scramble_range[0]-1:scramble_range[1]])
    plt.xticks(range(scramble_range[0], scramble_range[1]+1))
    plt.xlabel('scramble')
    plt.ylabel('# of solved')
    plt.show()


if __name__ == "__main__":

    # file_path = 'weights_MC.h5'
    # model = get_model()
    # model.load_weights(file_path)
    # for _ in range(2):
    #     seq, seq_str = random_sequence(15, metric='HTM')
    #     success, solution = astar(seq, model)
    #
    #     print('\n', seq_str, ': problem, length = ', len(seq))
    #     print(seqid2str(invseq(seq)), ': Inverse')
    #     print(solution, ': solved' if success else ': failed')

    evaluate(astar, scramble_range=(1, 20), N_TEST=100)
    # c = cube_env.Cube()
    # v = model.predict(np.array([c.get_state()])).ravel()[0]
    # print(v)
    # state_next, _ = transitions(c.get_state())
    # V = model.predict(np.array(state_next)).ravel()
    # print(V)
    # c.scramble(seqstr2id('RURULDLDLDFUFUFU'))
    # v = model.predict(np.array([c.get_state()])).ravel()[0]
    # print(v)
    # for idx, a in enumerate(action_list['HTM']):
    #     c.rotate(a)
    #     v = model.predict(np.array([c.get_state()])).ravel()[0]
    #     print(a, v)
    #     state_next, _ = transitions(c.get_state())
    #     V = model.predict(np.array(state_next)).ravel()
    #     print(V)
    #     c.rotate(invseq([idx])[0])
    # [100, 100, 100, 98, 89, 72, 48, 31, 15, 6, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0]