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

    collectionDict = {}


    # group all records for one velocity

    for init_vel in data:

        if str(int(init_vel[2]))+","+str(int(init_vel[3])) in collectionDict.keys():
            collectionDict_local = {}

            collectionDict_local = collectionDict[str(int(init_vel[2]))+","+str(int(init_vel[3]))].copy()

            collectionDict_local[str(int(init_vel[0]))+","+str(int(init_vel[1]))] = init_vel[4:]

            collectionDict[str(int(init_vel[2]))+","+str(int(init_vel[3]))] = collectionDict_local

            #print("Done", str(int(item[0]))+","+str(int(item[1])), str(int(item[2]))+","+str(int(item[3])))
        else:
            collectionDict_local = {}
            collectionDict_local[str(int(init_vel[0]))+","+str(int(init_vel[1]))] = init_vel[4:]
            collectionDict[str(int(init_vel[2]))+","+str(int(init_vel[3]))] = collectionDict_local


    lookupDict = {}


    for init_vel in collectionDict:

        for init_act in collectionDict[init_vel]:

            for point in range(end_point):

                pos_x = collectionDict[init_vel][init_act][point*2 +0]
                pos_y = collectionDict[init_vel][init_act][point*2 +1]

                pos_x, pos_y = centered_roundOff(pos_x, pos_y)


                if -4 <= pos_x <= 4 and -4 <= pos_y <= 4:

                    # if record already there then get it for update
                    if init_vel in lookupDict.keys():

                        lookupRecord = lookupDict[init_vel]

                    # else create new
                    else:

                        lookupRecord = np.zeros((9, 9, 2))



                    init_act_xy = init_act.split(",")
                    init_vel_xy = init_vel.split(",")


                    # if the action to reach a cell is (0,0) then update it directly
                    if lookupRecord[pos_x+4][pos_y+4][0] == 0 and lookupRecord[pos_x+4][pos_y+4][1] == 0:

                        lookupRecord[pos_x+4][pos_y+4][0] = int(init_act_xy[0])
                        lookupRecord[pos_x+4][pos_y+4][1] = int(init_act_xy[1])

                    # else find the distance of this velocity to current velocity
                    # and new velocity
                    # if new velocity has less distance
                    # replace else leave it
                    else:

                        dist_saved = math.sqrt( ( math.pow(lookupRecord[pos_x+4][pos_y+4][0] - int(init_vel_xy[0]), 2)
                                                  + math.pow(lookupRecord[pos_x+4][pos_y+4][1] - int(init_vel_xy[1]), 2) ) )

                        dist_new = math.sqrt( ( math.pow(int(init_act_xy[0]) - int(init_vel_xy[0]), 2) +
                                                math.pow(int(init_act_xy[1]) - int(init_vel_xy[1]), 2) ) )

                        if dist_saved > dist_new:
                            lookupRecord[pos_x+4][pos_y+4][0] = int(init_act_xy[0])
                            lookupRecord[pos_x+4][pos_y+4][1] = int(init_act_xy[1])




                    lookupDict[init_vel] = lookupRecord




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

    if not os.path.exists(lookupfile):
        buildLookupTable()