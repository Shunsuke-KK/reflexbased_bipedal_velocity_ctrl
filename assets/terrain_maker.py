import numpy as np
from PIL import Image
import os
import random


'''
plane -> up slope -> plane -> down slope -> plane -> step
'''
# img_black = np.zeros((200,100),np.uint8)
# img_white = np.ones((200,100),np.uint8)*255

def consective():
    floor_1 = np.zeros((100,100),np.uint8)
    slope_up = np.zeros((100,100),np.uint8)
    for i in range(slope_up.shape[1]):
        slope_up[:,i:] += 1
    floor_2 = np.ones((100,100),np.uint8)*100
    slope_down = np.ones((100,100),np.uint8)
    for i in range(slope_down.shape[1]):
        slope_down[:,i] = slope_up[:,-(i+1)]
    floor_3 = np.zeros((100,100),np.uint8)

    floor_4 = np.zeros((100,20),np.uint8)
    slope_up2 = np.zeros((100,30),np.uint8)
    for i in range(slope_up2.shape[1]):
        slope_up2[:,i:] += 1
    floor_5 = np.ones((100,15),np.uint8)*30
    slope_down2 = np.zeros((100,15),np.uint8)
    for i in range(slope_down2.shape[1]):
        slope_down2[:,i] = slope_up2[:,-(i+1)]
    rough_terrain = np.zeros((100,200),np.uint8)
    for i in range(rough_terrain.shape[1]):
        rdm = random.random()
        probability = 0.2
        if 0<=rdm<probability:
            dy = +1
        elif probability<=rdm<2*probability:
            dy = -1
        else:
            dy = 0

        if i == 0:
            rough_terrain[:,0] += slope_down2[0,-1]
        else:
            rough_terrain[:,i] = rough_terrain[:,i-1] + dy
            # print(rough_terrain[:,i])
            rough_terrain[:,i][rough_terrain[:,i]<5] = 5
            rough_terrain[:,i][rough_terrain[:,i]>15] = 15
            # rough_terrain[:,i] = max(0, rough_terrain[:,i])
            # rough_terrain[:,i] = min(rough_terrain[:,i], 100)

    # print(slope_down)
    img_1 = np.hstack([floor_1, slope_up, floor_2, slope_down, floor_3])
    img_2 = np.hstack([floor_4, slope_up2, floor_5,slope_down2, rough_terrain])
    # print(img_black)
    path =  os.getcwd()+'/terrain.png'
    path2 =  os.getcwd()+'/terrain2.png'
    pil_img_1 = Image.fromarray(img_1)
    pil_img_1.save(path)
    pil_img_2 = Image.fromarray(img_2)
    pil_img_2.save(path2)
    # cv2.imwrite(path, img_white)



def rough_terrain():
    # x=50[m], y=1[m]
    x = 40
    # kankaku = 0.5 # [m]
    mitsudo = 100
    terrain = np.zeros((int(30*6),int(x*mitsudo)),np.uint8)
    # for i in range(terrain.shape[0]):
    #     for j in range(terrain.shape[1]):
    #         if (i+j)%2==0:
    #             terrain[i][j] = 100
    for i in range(terrain.shape[1]):
        if i%mitsudo<mitsudo/2:
            terrain[:,i] = 100
    print(len(terrain[1]))
    terrain[:,:1000] = 0
    # for j in range(terrain.shape[0]):
    #     if j%10<10/2:
    #         terrain[j,:] = 100
    print(terrain)
    path =  os.getcwd()+'/rough_terrain.png'
    pil_img = Image.fromarray(terrain)
    pil_img.save(path)


rough_terrain()