import numpy as np

import h5py
from gym_unity.envs.unity_env import UnityEnv


def sampleTrajectory():

    action_repeat = 60
    action_range_around_zero = 10 # should be even
    period = 1

    if action_range_around_zero%2 != 0:
        return False

    env = UnityEnv(
        "/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_fixed_camera/Build/mazeContinuousTarget_fixed_camera_blank_board",
        0,
        use_visual=True,
        uint8_visual=True)

    list_of_data = []

    for i in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # velocity X [-5, -3, -1, 1, 3, 5]
        for j in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # velocity Y [-5, -3, -1, 1, 3, 5]
            for k in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # action X [-5, -3, -1, 1, 3, 5]
                for l in range(int(-1*action_range_around_zero/2), int(action_range_around_zero/2+1), period):  # action Y [-5, -3, -1, 1, 3, 5]

                    single_tuple = np.zeros(4+2*action_repeat)

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