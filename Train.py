import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from cube_env import *
from tqdm import tqdm
import csv
import os


N_SAMPLES = 200
N_EPOCH = 10000
gamma = 1

# https://androidkt.com/when-use-categorical_accuracy-sparse_categorical_accuracy-in-keras/
def sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


# Generate Keras DNN model
# modified from https://github.com/CVxTz/rubiks_cube and https://arxiv.org/abs/1805.07470
def get_model(lr=0.0001):
    input1 = Input((40,))

    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(1024)
    d4 = Dense(50)

    x1 = d1(input1)
    x1 = LeakyReLU()(x1)
    x1 = d2(x1)
    x1 = LeakyReLU()(x1)
    x1 = d3(x1)
    x1 = LeakyReLU()(x1)
    x1 = d4(x1)
    x1 = LeakyReLU()(x1)

    out_value = Dense(1, activation="linear", name="value")(x1)
    out_policy = Dense(len(action_list['HTM']), activation="softmax", name="policy")(x1)

    model = Model(input1, [out_value, out_policy])

    model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                  metrics={"policy": sparse_categorical_accuracy})
    model.summary()

    return model


def train():
    file_path = 'checkpoint-batch-{}-trial-001.h5'.format(N_SAMPLES)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)
    callbacks_list = [checkpoint, early, reduce_on_plateau]

    model = get_model(lr=0.0001)
    if os.path.exists(file_path):
        model.load_weights(file_path)

    for i in tqdm(range(N_EPOCH)):
        history = []  # empty the log list every epoch after updating logfile
        states = []
        cube_depth = []
        for j in range(N_SAMPLES):
            _cubes_state, _cube_depth = generate_episode(20)  # gen sequence gives [[state_1 to state_20], [0 to 20]]
            states.extend(_cubes_state)                  # size(_cube_state) = 40
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
        for _ in range(20):  # update w~ after 20 steps

            replayBuffer_target_v = []
            replayBuffer_target_policy = []

            # calculate v(s') for all a for all s
            v_state_next, _ = model.predict(np.array(state_next), batch_size=1024)
            v_state_next = v_state_next.ravel().tolist()
            # group the v(s') for s
            v_state_next = list(chunker(v_state_next, size=len(action_list['HTM'])))
            # v_state_next = [[v(s'(s1,a1)),...,v(s'(s1,a_n))], [v(s'(s2,a1)),...,v(s'(s2,a_n))], ...]
            #                 list of possible v(s') for all a corresponding to each state

            for s, r, v_s_next in zip(states, rewards, v_state_next):
                q = np.array(r) + gamma * np.array(v_s_next)
                # Value Iteration
                target_v = np.max(q)
                target_policy = np.argmax(q)
                # Store in replay buffer
                replayBuffer_target_v.append(target_v)
                replayBuffer_target_policy.append(target_policy)

            # normalization
            replayBuffer_target_v = (replayBuffer_target_v-np.mean(replayBuffer_target_v))/\
                                    (np.std(replayBuffer_target_v)+0.0001)

            sample_weights = 1. / np.array(cube_depth)
            sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)

            result = model.fit(np.array(states),
                               [np.array(replayBuffer_target_v), np.array(replayBuffer_target_policy)[..., np.newaxis]],
                               epochs=1, batch_size=128, sample_weight=[sample_weights, sample_weights])
            # run only one epoch as it updates the value only once for every iteration
            history.append(result.history)
            # with open('result.csv', 'a', newline = '') as csv:
            #     writer = csv.DictWriter(csv, fieldnames=csv_columns)
            #     writer.writeheader()
            #     for data in dict_data:
            #         writer.writerow(data)

        with open('history1.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=history[0].keys())
            if not csvfile.tell():
                writer.writeheader()
            writer.writerows(history)

        model.save_weights(file_path)


if __name__ == "__main__":
    train()
