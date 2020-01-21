import sys
base_dir = '/homes/gkumar/curiosityDrivenGraphBuilding'
sys.path.append(base_dir)

import os
import numpy as np
import h5py
import math

np.set_printoptions(threshold=np.inf)

datafile = "../data_gen/data_11x11x11x11x60.h5"
lookupfile = "data_processed_11x11x11x11x60.h5"
lookupfile1 = "data_processed_1_11x11x11x11x60.h5"

#datafile = "../data_gen/data_6x6x6x6x150.h5"
#lookupfile = "data_processed_6x6x6x6x150.h5"

end_point = 60





def readTrajectory():

    # open data base file
    h5f =h5py.File(datafile, "r")

    # read in a list
    verification_data = h5f['dataset_1'][...]

    # close file
    h5f.close()

    # return read data
    return verification_data





def buildLookupTable():

    data = readTrajectory()

    lookupDict = {}

    for item in data:

        for point in range(end_point):

            pos_x = item[3 + point*2 +1]
            pos_y = item[3 + point*2 +2]

            pos_x, pos_y = centered_roundOff(pos_x, pos_y)




            # if record already there then get it for update
            if str(int(item[2]))+","+str(int(item[3])) in lookupDict.keys():

                lookupRecord = lookupDict[str(int(item[2]))+","+str(int(item[3]))]

            # else create new
            else:

                lookupRecord = np.zeros((9, 9, 2))




            # if the action to reach a cell is (0,0) then update it directly
            if lookupRecord[pos_x][pos_y][0] == 0 and lookupRecord[pos_x][pos_y][1] == 0:

                lookupRecord[pos_x][pos_y][0] = item[0]
                lookupRecord[pos_x][pos_y][1] = item[1]

            # else find the distance of this velocity to current velocity
            # and new velocity
            # if new velocity has less distance
            # replace else leave it
            else:

                dist_saved = math.sqrt( ( math.pow(lookupRecord[pos_x][pos_y][0] - item[2], 2)
                                          + math.pow(lookupRecord[pos_x][pos_y][1] - item[3], 2) ) )

                dist_new = math.sqrt( ( math.pow(item[0] - item[2], 2) + math.pow(item[1] - item[3], 2) ) )

                if dist_saved > dist_new:
                    lookupRecord[pos_x][pos_y][0] = item[0]
                    lookupRecord[pos_x][pos_y][1] = item[1]




            lookupDict[str(int(item[2]))+","+str(int(item[3]))] = lookupRecord




    list_of_data = np.zeros((len(lookupDict.keys()),9*9+1, 2))





    for i, key in enumerate(lookupDict):
        #print(i, key)
        keys = key.split(",")
        list_of_data[i][0][0] = keys[0]
        list_of_data[i][0][1] = keys[1]
        list_of_data[i][1:] = np.reshape(lookupDict[key],(81,2))

    h5f = h5py.File(lookupfile, 'w')
    h5f.create_dataset('dataset_1', data=list_of_data)
    h5f.close()





def buildLookupTable1():

    data = readTrajectory()

    lookupDict = {}

    for item in data:

        for point in range(end_point):

            pos_x = item[3 + point*2 +1]
            pos_y = item[3 + point*2 +2]

            pos_x, pos_y = centered_roundOff(pos_x, pos_y)




            # if record already there then get it for update
            if str(int(item[0]))+","+str(int(item[1])) in lookupDict.keys():

                lookupRecord = lookupDict[str(int(item[0]))+","+str(int(item[1]))]

            # else create new
            else:

                lookupRecord = np.zeros((9, 9, 2))




            # if the action to reach a cell is (0,0) then update it directly
            if lookupRecord[pos_x][pos_y][0] == 0 and lookupRecord[pos_x][pos_y][1] == 0:

                lookupRecord[pos_x][pos_y][0] = item[2]
                lookupRecord[pos_x][pos_y][1] = item[3]

            # else find the distance of this velocity to current velocity
            # and new velocity
            # if new velocity has less distance
            # replace else leave it
            else:

                dist_saved = math.sqrt( ( math.pow(lookupRecord[pos_x][pos_y][0] - item[0], 2)
                                          + math.pow(lookupRecord[pos_x][pos_y][1] - item[1], 2) ) )

                dist_new = math.sqrt( ( math.pow(item[2] - item[0], 2) + math.pow(item[3] - item[1], 2) ) )

                if dist_saved > dist_new:
                    lookupRecord[pos_x][pos_y][0] = item[2]
                    lookupRecord[pos_x][pos_y][1] = item[3]




            lookupDict[str(int(item[0]))+","+str(int(item[1]))] = lookupRecord




    list_of_data = np.zeros((len(lookupDict.keys()),9*9+1, 2))





    for i, key in enumerate(lookupDict):
        #print(i, key)
        keys = key.split(",")
        list_of_data[i][0][0] = keys[0]
        list_of_data[i][0][1] = keys[1]
        list_of_data[i][1:] = np.reshape(lookupDict[key],(81,2))

    h5f = h5py.File(lookupfile1, 'w')
    h5f.create_dataset('dataset_1', data=list_of_data)
    h5f.close()






def centered_roundOff(pos_x, pos_y):
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

    return pos_x, pos_y





if __name__ == '__main__':

    #if not os.path.exists(lookupfile):
    #    buildLookupTable()

    if not os.path.exists(lookupfile1):
        buildLookupTable1()