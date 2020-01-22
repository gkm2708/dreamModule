import numpy as np

import h5py
from gym_unity.envs.unity_env import UnityEnv
import math

def sampleTrajectory():

    action_repeat = 300

    action_range_around_zero = 20 # should be even

    #action_range_around_zero = [-9, -8, -7, -6, -5, 0, 5, 6, 7, 8, 9]

    period = 1

    if action_range_around_zero%2 != 0:
        return False

    env = UnityEnv(
        "/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_fixed_camera_data_collection/Build/mazeContinuousTarget_fixed_camera_data_collection",
        0, use_visual=True, uint8_visual=True)

    list_of_data = []

    for i in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # velocity X [-5, -3, -1, 1, 3, 5]
        for j in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # velocity Y [-5, -3, -1, 1, 3, 5]
            print(i,j)
            for k in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # action X [-5, -3, -1, 1, 3, 5]
                for l in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # action Y [-5, -3, -1, 1, 3, 5]

                    single_tuple = np.zeros(4+2*action_repeat)

                    obs_fovea = env.reset()
                    obs_fovea_next, reward, done, info = env.step([[i], [j], [k], [l]])

                    # action
                    single_tuple[0] = i
                    single_tuple[1] = j
                    # velocity
                    single_tuple[2] = k
                    single_tuple[3] = l

                    for m in range(0,action_repeat):

                        single_tuple[3 + m*2 + 1] = info["brain_info"].vector_observations[0][2]
                        single_tuple[3 + m*2 + 2] = info["brain_info"].vector_observations[0][3]

                        x_vel_new = info["brain_info"].vector_observations[0][6]
                        y_vel_new = info["brain_info"].vector_observations[0][7]

                        if math.sqrt(
                                math.pow(
                                    (single_tuple[3 + m*2 + 1] - single_tuple[4]),2)
                                + math.pow(
                                    (single_tuple[3 + m*2 + 2] - single_tuple[5]),2)) < 6:

                            obs_fovea_next, reward, done, info = env.step([[i], [j], [x_vel_new], [y_vel_new]])

                        else:

                            for n in range(m, action_repeat):
                                single_tuple[3 + n * 2 + 1] = single_tuple[3 + (m-1) * 2 + 1]
                                single_tuple[3 + n * 2 + 2] = single_tuple[3 + (m-1) * 2 + 2]

                            break


                    list_of_data.append(single_tuple)

    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('dataset_1', data=list_of_data)
    h5f.close()