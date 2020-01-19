import matplotlib.pyplot as plt
import numpy as np

def mergeImages():

    for i in range(-5, 6, 2):  # velocity X [-5, -3, -1, 1, 3, 5]
        for j in range(-5, 6, 2):  # velocity Y [-5, -3, -1, 1, 3, 5]
            name_string = str(float(i)) + "_" + str(float(j))
            data00 = plt.imread("/homes/gkumar/GenModel_nextStateAction/results/result_only_velocity/sigmoid_categorical_crossentropy/predictions_scaled/prediction_"+name_string+".png", 'r')
            data01 = plt.imread("/homes/gkumar/GenModel_nextStateAction/results/result_only_velocity/sigmoid_binary_crossentropy/predictions_scaled/prediction_"+name_string+".png", 'r')
            data02 = plt.imread("/homes/gkumar/GenModel_nextState_2/ground_truth_3/ground_truth_"+name_string+".png", 'r')
            data10 = plt.imread("/homes/gkumar/GenModel_nextStateAction/results/result_only_velocity/softmax_categorical_crossentropy/predictions_scaled/prediction_"+name_string+".png", 'r')
            data11 = plt.imread("/homes/gkumar/GenModel_nextStateAction/results/result_only_velocity/softmax_binary_crossentropy/predictions_scaled/prediction_"+name_string+".png", 'r')
            data12 = plt.imread("/homes/gkumar/GenModel_nextState_2/ground_truth_4/ground_truth_"+name_string+".png", 'r')


            finalImage1 = np.concatenate([data01[:,:,0], data11[:,:,0], data02[:,-81:,0]], axis=1)

            finalImage2 = np.concatenate([data00[:,:,0], data10[:,:,0], data12[:,-81:,0]], axis=1)

            plt.imsave("concatenated_result/image_"+name_string+".png", np.concatenate([finalImage1, finalImage2], axis=0), cmap='gray')

            print("done")

