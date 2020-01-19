import numpy as np
import os
import h5py
from learn import Learner
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.inf)

action_repeat = 150
action_repeat_length = 30

max_epoch = 30
best_accuracy = -0.1

train = True
test = True

datafile = "../data_gen/data_1296.h5"

ll_activation = "sigmoid"
#loss_fun = "categorical_crossentropy"
loss_fun = "binary_crossentropy"

learner = Learner(ll_activation, loss_fun)



def readTrajectory():
    # open data base file
    h5f =h5py.File(datafile, "r")
    # read in a list
    verification_data = h5f['dataset_1'][...]
    # close file
    h5f.close()
    # return read data
    return verification_data


def plotSimple():
    learner.load_model()
    # randomly sample a few input values and predict output
    for i in range(-5, 6, 2):
        for j in range(-5, 6, 2):
            for k in range(-5, 6, 2):
                for l in range(-5, 6, 2):
                    input = np.zeros((1,4))
                    input[0][0] = i
                    input[0][1] = j
                    input[0][2] = k
                    input[0][3] = l
                    learner.plot_prediction(str(i)+"_"+str(j)+"_"+str(k)+"_"+str(l), input)
    print("End")



def plotAmplified():
    learner.load_model()

    full_result = []

    for i in range(-5, 6, 2):
        for j in range(-5, 6, 2):
            for k in range(-5, 6, 2):
                for l in range(-5, 6, 2):
                    input = np.zeros((1,4))
                    input[0][0] = i
                    input[0][1] = j
                    input[0][2] = k
                    input[0][3] = l

                    result = np.zeros(85)

                    result[0] = i
                    result[1] = j
                    result[2] = k
                    result[3] = l
                    result[4:] = learner.model.predict(input)

                    full_result.append(result)

    full_result_np = np.asarray(full_result)
    full_result_np = full_result_np[:,4:]
    full_result_np = full_result_np.flatten()
    full_result_np = np.sort(full_result_np)

    median = (full_result_np[int(full_result_np.shape[0]/2-1)] + full_result_np[int(full_result_np.shape[0]/2)] )/2

    full_result_min = np.min(full_result_np)
    full_result_max = np.max(full_result_np)

    f = open(ll_activation+"_"+loss_fun+"/predictions_scaled/_info_"+\
             str(round(full_result_min,3))+\
             "_"+str(round(full_result_max,3))+\
             "_"+str(round(median,3))+".txt", "w")

    f.write("\nmedian : "+str(median))
    f.write("\nmedian*1.5 : "+str(median*1.5))
    f.write("\nmax : "+str(full_result_max))
    f.write("\nmin : "+str(full_result_min))

    for item in full_result:

        title = str(item[0])+"_"+str(item[1])+"_"+str(item[2])+"_"+str(item[3])

        f.write("\nsum " + title + " : " + str(np.sum(item[4:])))

        result_modified = np.reshape(np.asarray(item[4:]*255/median), (9,9))
        n = 9
        result_expanded = np.kron(result_modified, np.ones((n, n)))


        plt.title(title)
        plt.imsave(ll_activation+"_"+loss_fun+'/predictions_scaled/prediction_' + title + '.png', result_expanded, cmap='gray')
        plt.close()

    f.write("\nsorted all : " + str(full_result_np))
    f.close()


if __name__ == '__main__':


    if not os.path.exists(ll_activation+"_"+loss_fun):
        os.makedirs(ll_activation+"_"+loss_fun)
        os.makedirs(ll_activation+"_"+loss_fun+"/predictions")
        os.makedirs(ll_activation+"_"+loss_fun+"/predictions_scaled")
        os.makedirs(ll_activation+"_"+loss_fun+"/samples")


    if train:

        # read dataset
        data = readTrajectory()

        # !!!!!!! Reshape data here
        input = data[:,0:4]
        label_data = data[:,3 + action_repeat_length*2 +1:3 + action_repeat_length*2 +3]

        label = []

        for j in range(len(label_data)):

            # what label -we need to find
            pos_x = label_data[j][0]
            pos_y = label_data[j][1]

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

            true_label = np.zeros((9, 9))

            if 0 <= pos_x + 4 < 9:
                pos_x = pos_x + 4
            else:
                pos_x = -1

            if 0 <= pos_y + 4 < 9:
                pos_y = pos_y + 4
            else:
                pos_y = -1

            if pos_x >= 0 and pos_y >= 0:
                true_label[pos_x][pos_y] = 1.0

            label.append(true_label.flatten())

        # loop over for epoch
        label = np.asarray(label)

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
        plotSimple()
        plotAmplified()