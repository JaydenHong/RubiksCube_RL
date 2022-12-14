import cube_env
from cube_env import *
from Train import get_model, N_SAMPLES


def main():
    # Read weights
    file_path = 'test.h5'
    # file_path = 'checkpoint-batch-{}-trial-001.h5'.format(N_SAMPLES)
    model = get_model()
    model.load_weights(file_path)

    # Gen Rand state
    c = cube_env.Cube()
    seq, seq_str = random_sequence(10, 'HTM')
    states = c.scramble(seq)
    state = states[-1]

    existing_cubes = set()

    for _ in range(1000):
        value, policy = model.predict(np.array(state), batch_size=1024)

        new_list_sequences = []

        for x, policy in zip(list_sequences, policy):
            new_sequences = [x + [x[-1].copy()(action)] for action in action_list['HTM']]

            pred = np.argsort(policy)

            cube_1 = x[-1].copy()(Actionlist[pred[-1]])
            cube_2 = x[-1].copy()(Actionlist[pred[-2]])

            new_list_sequences.append(x + [cube_1])
            new_list_sequences.append(x + [cube_2])

        print("new_list_sequences", len(new_list_sequences))
        last_states_flat = [flatten_1d_b(x[-1]) for x in new_list_sequences]
        value, _ = model.predict(np.array(last_states_flat), batch_size=1024)
        value = value.ravel().tolist()
        for x, v in zip(new_list_sequences, value):
            x[-1].score = v if str(x[-1]) not in existing_cubes else -1

        new_list_sequences.sort(key=lambda x: x[-1].score, reverse=True)

        new_list_sequences = new_list_sequences[:100]

        existing_cubes.update(set([str(x[-1]) for x in new_list_sequences]))

        list_sequences = new_list_sequences

        list_sequences.sort(key=lambda x: perc_solved_cube(x[-1]), reverse=True)

        prec = perc_solved_cube((list_sequences[0][-1]))

        print(prec)

        if prec == 1:
            break

    print(perc_solved_cube(list_sequences[0][-1]))
    print(list_sequences[0][-1])


if __name__ == "__main__":
    main()
