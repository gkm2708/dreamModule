import os
import numpy as np
import math

import h5py
from gym_unity.envs.unity_env import UnityEnv
from learn import Learner

np.set_printoptions(threshold=np.inf)

action_repeat = 150
action_repeat_length = 30

max_epoch = 3000
best_accuracy = -0.1

train = False
test = True

datafile = "data_1296.h5"


learner = Learner()

if train:
    # assume that unity reads that file and generates maze dynamically
    env = UnityEnv("/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_fixed_camera/Build/mazeContinuousTarget_fixed_camera_blank_board",
                   0,
                   use_visual=True,
                   uint8_visual=True)



def drawTrajectory():

    list_of_data = []

    for i in range(-5, 6, 5):
        for j in range(-5, 6, 5):
            for k in range(-5, 6, 5):
                for l in range(-5, 6, 5):
                    single_tuple = np.zeros(304)

                    obs_fovea = env.reset()
                    obs_fovea_next, reward, done, info = env.step([[i], [j], [k], [l]])

                    single_tuple[0] = i
                    single_tuple[1] = j
                    single_tuple[2] = k
                    single_tuple[3] = l

                    for m in range(0,action_repeat):

                        single_tuple[3 + m*2 + 1] = info["brain_info"].vector_observations[0][2]
                        single_tuple[3 + m*2 + 2] = info["brain_info"].vector_observations[0][3]

                        x_vel_new = info["brain_info"].vector_observations[0][6]
                        y_vel_new = info["brain_info"].vector_observations[0][7]
                        obs_fovea_next, reward, done, info = env.step([[i], [j], [x_vel_new], [y_vel_new]])

                    list_of_data.append(single_tuple)

    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('dataset_1', data=list_of_data)
    h5f.close()


def readTrajectory():
    # open data base file
    h5f =h5py.File(datafile, "r")
    # read in a list
    verification_data = h5f['dataset_1'][...]
    # close file
    h5f.close()
    # return read data
    return verification_data




if __name__ == '__main__':


    # if no dataset
    if not os.path.exists(datafile):

        #   prepare one
        drawTrajectory()

    # if dataset
    else:

        # read dataset
        data = readTrajectory()

        # !!!!!!! Reshape data here
        #input = data[:,0:4]
        #label_data = data[:,3 + action_repeat_length*2 +1:3 + action_repeat_length*2 +3]

        # label action
        label = data[:,2:4]

        # input the current tilt
        input1 = data[:,0:2]
        # and final position
        input2_data = data[:,3 + action_repeat_length*2 +1:3 + action_repeat_length*2 +3]
        input2 = []

        for j in range(len(input2_data)):

            # what label -we need to find
            pos_x = input2_data[j][0]
            pos_y = input2_data[j][1]


            pos_x_int = int(pos_x)

            lower_limit = pos_x_int - 0.5
            upper_limit = pos_x_int + 0.5

            if lower_limit <= pos_x <= upper_limit:
                pos_x = pos_x_int
            elif pos_x < lower_limit:
                pos_x = pos_x_int - 1
            elif pos_x > upper_limit:
                pos_x = pos_x_int + 1


            pos_y_int = int(pos_y)

            lower_limit = pos_y_int - 0.5
            upper_limit = pos_y_int + 0.5

            if lower_limit <= pos_y <= upper_limit:
                pos_y = pos_y_int
            elif pos_y < lower_limit:
                pos_y = pos_y_int - 1
            elif pos_y > upper_limit:
                pos_y = pos_y_int + 1

            input2.append([pos_x, pos_y])

        # loop over for epoch

        input2 = np.asarray(input2)

        input = np.concatenate([input1, input2], axis=1)


        if train:

            for i in range(max_epoch):

                # fit model for this epoch
                current_accuracy = learner.learn_step(input, label)

                # plot loss and accuracy
                learner.plot_loss_graph()
                learner.plot_accuracy_graph()

                #   check accuracy
                if current_accuracy > best_accuracy:
                    # save model
                    learner.save_model()
                    best_accuracy = current_accuracy

                if current_accuracy == 1.0:
                    break
                    learner.save_model()

        if test:
            f = open("result.txt", "a")
            learner.load_model()
            for item in input:

                final = np.zeros((1,4))
                final[0][0] = item[0]
                final[0][1] = item[1]
                final[0][2] = item[2]
                final[0][3] = item[3]

                #prediction = learner.plot_prediction(final)


                f.write( str(item) + ">" + str(learner.plot_prediction(final)) + "\n" )

            f.close()
    print("End")

