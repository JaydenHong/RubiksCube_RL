import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from cube_env import *
from tqdm import tqdm
import csv
import os


# Generate Keras DNN model
# modified from https://github.com/CVxTz/rubiks_cube and https://arxiv.org/abs/1805.07470
def get_model(lr=0.0001):
    input1 = Input((40,))

    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(512)
    d4 = Dense(128)

    x1 = d1(input1)
    x1 = LeakyReLU()(x1)
    x1 = d2(x1)
    x1 = LeakyReLU()(x1)
    x1 = d3(x1)
    x1 = LeakyReLU()(x1)
    x1 = d4(x1)
    x1 = LeakyReLU()(x1)

    out_value = Dense(1, activation="linear", name="value")(x1)

    model = Model(input1, out_value)

    model.compile(loss="mse", optimizer=Adam(lr))
    model.summary()

    return model


N_SAMPLES = 800
N_EPOCH = 100
gamma = 1

def train():
    file_path_load = 'weights_dqn.h5'
    file_path_save = 'weights_dqn.h5'
    checkpoint = ModelCheckpoint(file_path_load, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)
    callbacks_list = [checkpoint, early, reduce_on_plateau]

    model = get_model(lr=0.0001)
    if os.path.exists(file_path_load):
        model.load_weights(file_path_load)

    for i in range(tqdm(N_EPOCH)):
        print('Epoch:', i)
        history = []  # empty the log list every epoch after updating logfile
        states = []
        cube_depth = []
        for j in range(N_SAMPLES):
            _cubes_state, _cube_depth = generate_episode(20)  # gen sequence gives [[state_1 to state_20], [0 to 20]]
            states.extend(_cubes_state)
            cube_depth.extend(_cube_depth)

        # states = 20 states * N_SAMPLES = list of 2000 states
        # cube_depth =  [1 to 20] * N_SAMPLES = list of cube depth corresponding to each state

        rewards = []
        state_next = []

        for s in states:
            s_next, r = transitions(s)  # gives s', r for all a
            rewards.append(r)
            state_next.extend(s_next)
        # rewards=[[r for all a from s1], [r for all a from s2], ...]
        # state_next=[[s' for all a form s1], [s-' for all a form s2],...]
        state_next_ref = list(chunker(state_next, size=len(action_list['HTM'])))

        for _ in range(20):  # every 20 iteration, regenerate the replay memory

            replayBuffer_target_v = []

            # calculate v(s') for all a for all s
            v_state_next = model.predict(np.array(state_next), batch_size=1024)
            v_state_next = v_state_next.ravel().tolist()

            # group the v(s') for s
            v_state_next = list(chunker(v_state_next, size=len(action_list['HTM'])))
            # v_state_next = [[v(s'(s1,a1)),...,v(s'(s1,a_n))], [v(s'(s2,a1)),...,v(s'(s2,a_n))], ...]
            #                 list of possible v(s') for all a corresponding to each state
            for s, r, s_next, v_s_next in zip(states, rewards, state_next_ref, v_state_next):
                if state_Terminal in s_next:
                    target_v = -1
                else:
                    y = np.array(r) + gamma * np.array(v_s_next)
                    target_v = np.max(y)
                # target_policy = np.argmax(q)
                # Store in replay buffer
                replayBuffer_target_v.append(target_v)
                # replayBuffer_target_policy.append(target_policy)

            # # normalization
            # replayBuffer_target_v = (replayBuffer_target_v-np.mean(replayBuffer_target_v))/\
            #                         (np.std(replayBuffer_target_v)+0.0001)

            # sample_weights = 1. / np.array(cube_depth)
            # sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)
            if i < 100:
                result = model.fit(np.array(states), np.array(cube_depth)*-1, epochs=1, batch_size=512)
            else:
                result = model.fit(np.array(states), np.array(replayBuffer_target_v), epochs=1, batch_size=512)
            # i.e. policy and value are updated every iteration
            history.append(result.history)

        with open('history_value_only.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=history[0].keys())
            if not csvfile.tell():
                writer.writeheader()
            writer.writerows(history)

        model.save_weights(file_path_save)


if __name__ == "__main__":
    train()
