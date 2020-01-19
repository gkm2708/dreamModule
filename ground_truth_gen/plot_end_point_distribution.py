import numpy as np
import h5py
import matplotlib.pyplot as plt


# create grid and write to file
maze_trials, max_episode, max_steps = 1, 100000, 900
action_repeat = 30




def plotEndPointDistribution():

    # verify
    h5f =h5py.File("../data_gen/data_1296.h5", "r")
    verification_data = h5f['dataset_1'][...]
    h5f.close()

    endpoint_data = np.zeros((36,37,2))

    for i in range(36):
        endpoint_data[i,0,0] = verification_data[i*36,0]
        endpoint_data[i,0,1] = verification_data[i*36,1]
        for j in range(36):

            pos_x_int = int(verification_data[i*36+j, 3 + 30*2 +1])

            lower_limit = pos_x_int - 0.5
            upper_limit = pos_x_int + 0.5

            if lower_limit <= verification_data[i*36+j, 3 + 30*2 +1] <= upper_limit:
                endpoint_data[i,j+1,0] = pos_x_int
            elif verification_data[i*36+j, 3 + 30*2 +1] < lower_limit:
                endpoint_data[i,j+1,0] = pos_x_int - 1
            elif verification_data[i*36+j, 3 + 30*2 +1] > upper_limit:
                endpoint_data[i,j+1,0] = pos_x_int + 1

            pos_y_int = int(verification_data[i*36 +j,3 + 30*2 +2])

            lower_limit = pos_y_int - 0.5
            upper_limit = pos_y_int + 0.5

            if lower_limit <= verification_data[i*36 +j,3 + 30*2 +2] <= upper_limit:
                endpoint_data[i,j+1,1] = pos_y_int
            elif verification_data[i*36 +j,3 + 30*2 +2] < lower_limit:
                endpoint_data[i,j+1,1] = pos_y_int - 1
            elif verification_data[i*36 +j,3 + 30*2 +2] > upper_limit:
                endpoint_data[i,j+1,1] = pos_y_int + 1


    limit = 4
    multiplier = int(255/(limit-1))

    for i in range(36):
        image_data = np.zeros((9, 9))
        for j in range(36):
            image_data[int(endpoint_data[i,j+1,0]+4), int(endpoint_data[i,j+1,1]+4)] += 1

        image_data = np.asarray([[item if item < limit else limit-1 for item in row] for row in image_data])


        image_data = np.asarray([[(item)*multiplier for item in row] for row in image_data])

        n = 9
        result_expanded = np.kron(image_data, np.ones((n, n)))

        name_string = str(endpoint_data[i,0,0]) +"_"+ str(endpoint_data[i,0,1])

        plt.title( name_string )
        plt.imsave('ground_truth_'+str(limit)+'/ground_truth_'+name_string+'.png', result_expanded, cmap='gray')
        plt.close()
        print("Done")

    print("Done")
