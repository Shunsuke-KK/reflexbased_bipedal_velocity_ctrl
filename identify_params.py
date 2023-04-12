import os
import pickle
from custom_env import Reflex_WALK_Env
from reflex_opt.run_policy_identify import run_policy_identify
import numpy as np

import pandas as pd
from pandas import DataFrame
import math

previous_data = True
previous_data = False
if not previous_data:
    forder_name = 'opt_multi_27'
    save_path = '/home/shunsuke/cpg/reflex_opt/save_data'
    save_folder = save_path + '/' + forder_name


    def l_s_method_original(x, y, weight, degree):
        t = y #tベクトルを作成
        t = t*weight
        phi = DataFrame()
        #Φ行列を作成
        for i in range(0, degree+1):
            p = x ** i
            p = p*weight
            p.name = "x ** %d" % i
            phi = pd.concat([phi, p], axis = 1)

        #w=(Φ^T*Φ)^(−1)*Φ^T*tを計算
        ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi).astype(np.float64)), phi.T), t)
        #関数f(x)を作成
        def f(x):
            y = 0
            for i, w in enumerate(ws):
                y += w * (x ** i)
            return y

        return (f, ws)



    params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA',
            'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
            'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v',
            'vel','cot','value']
    data = {}
    optimizer = {}
    dataset = DataFrame(columns = params_label)
    num = 0
    while True:
    # for j in range(2):
        filename = f'vel_gen{num}.pickle'
        if os.path.exists(os.path.join(save_folder,filename)):
            with open(os.path.join(save_folder,filename), mode='rb') as g:
                # print(os.path.join(save_folder,filename))
                load_checkpoint = pickle.load(g)
                loaded_dataframe = load_checkpoint['DataFrame']
                data[f'{num}'] = loaded_dataframe
                dataset = pd.concat([dataset, loaded_dataframe], ignore_index = True)
                optimizer[f'{num}'] = load_checkpoint['optimizer']

            num += 1
        else:
            print(f'data num = {num-1}')
            break

    dataset=dataset[dataset['cot']<100]
    dataset = dataset.sort_values('vel')

    x = dataset['vel']
    print(f'n = {len(x)}')
    cot = dataset['cot']
    b = 30
    weight = np.empty(0)
    coefficient = 800
    for i in range(b):
        average = np.average(cot.iloc[:i+b])
        impact = math.pow(coefficient, (average-cot.iloc[i])/average)
        weight = np.append(weight,impact)
    for i in range(b,len(x)-b):
        average = np.average(cot.iloc[i-b:i+b])
        impact = math.pow(coefficient, (average-cot.iloc[i])/average)
        weight = np.append(weight,impact)
    for i in range(len(x)-b,len(x)):
        average = np.average(cot.iloc[i-b:])
        impact = math.pow(coefficient, (average-cot.iloc[i])/average)
        weight = np.append(weight,impact)
    print(weight)
    degree = 6
    # ws_store = np.empty((0,len(params_label)))
    columns = [f'x ** {i}' for i in range(0,degree+1)]
    ws_dataset = DataFrame(columns = columns)
    for label in params_label:
        f, ws = l_s_method_original(x, dataset[label], weight, degree)
        ws = DataFrame([ws], columns=columns)
        ws_dataset = pd.concat([ws_dataset, ws], ignore_index = True)
    ws_dataset = ws_dataset.set_axis(params_label, axis='index')
    print(ws_dataset)
    print(ws_dataset.loc['G_SOL'])

    ctrl_params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA',
            'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
            'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v',]


    with open(os.path.join(os.getcwd(),'function_A800_27.pickle'), mode='wb') as g:
            checkpoint = {'dataset' : dataset ,'ws_dataset' : ws_dataset}
            pickle.dump(checkpoint, g)









# if __name__ == '__main__':
if previous_data:
    import argparse
    parser = argparse.ArgumentParser()
    path=os.getcwd()+'/assets'

    VPenv = Reflex_WALK_Env(path=path)

    data_path = '/home/shunsuke/cpg/graph/code'
    # with open(os.path.join(data_path,'function_A1.pickle'), mode='rb') as g:
    with open('function_A1_31.pickle', mode='rb') as g:
        load_checkpoint = pickle.load(g)
        ws_dataset_A1 = load_checkpoint['ws_dataset']
    # with open(os.path.join(data_path,'function_A800.pickle'), mode='rb') as g:
    with open('function_A800_31.pickle', mode='rb') as g:
        load_checkpoint = pickle.load(g)
        ws_dataset_A800 = load_checkpoint['ws_dataset']

    ctrl_params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA',
                'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'phi_h_off_SP', 'phi_k_off_SP','k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
                'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'c_d', 'c_v',]

    # print(ws_dataset)
    # label = ctrl_params_label[0]
    # ws_dataset.loc[label] = ws_dataset2.loc[label]
    def f(x, ws):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    tar_vel = [0.5+i*0.1 for i in range(10)]
    # tar_vel[0] = 0.505
    # tar_vel[-1] = 1.43
    # tar_vel[-2] = 1.25
    # tar_vel = [0.52,0.532,0.7,0.83,0.9,1.0,1.16,1.2,1.25,1.43]
    # tar_vel[2]=0.63
    # none 0.95 0.8671656797262411 0.6315080219050813
    # none 1.15 1.1699361925941005 0.731643412662528
    # none 1.35 1.4150444770551667 0.883234785875106
    # none 1.45 1.2889569163257157 1.0556783063025195

    # tar_vel = [0.5,0.6,0.7,0.8,0.95,1.0,1.1,1.15,1.2,1.35] # 32
    # tar_vel = [0.95,1.15,1.35,1.45]
    print(tar_vel)
    previous_data = True
    previous_data = False
    if previous_data:
        with open(os.path.join('identify31.pickle'), mode='rb') as g:
            plot_data = pickle.load(g)
    else:
        plot_data = {}
    actual_velocity = np.empty(0)
    measured_cot = np.empty(0)
    
    if previous_data==False:
        for v in tar_vel:
            mean_vel, cot, flag = run_policy_identify(env=VPenv, ws_dataset=ws_dataset_A1, v=v, param='none')
            # if flag==False and int(v*10)!=15:
            if flag==False:
                mean_vel, cot, flag = run_policy_identify(env=VPenv, ws_dataset=ws_dataset_A1, v=v, change_noise=True, param='none')
                if flag == False:
                    while flag==False:
                        mean_vel, cot, flag = run_policy_identify(env=VPenv, ws_dataset=ws_dataset_A1, v=v, change_noise=True, change_noise2=True, param='none')
            actual_velocity = np.append(actual_velocity,mean_vel)
            measured_cot = np.append(measured_cot, cot)
        print('')
        plot_data['base']={'vel':actual_velocity, 'cot':measured_cot}



    for param in ctrl_params_label:
        actual_velocity = np.empty(0)
        measured_cot = np.empty(0)
        ws_dataset = ws_dataset_A1.copy()
        ws_dataset.loc[param] = ws_dataset_A800.loc[param]
        print(ws_dataset)
        for v in tar_vel:
            mean_vel, cot, flag = run_policy_identify(env=VPenv, ws_dataset=ws_dataset, v=v, param=param)
            # if flag==False and int(v*10)!=15:
            if flag==False:
                mean_vel, cot, flag = run_policy_identify(env=VPenv, ws_dataset=ws_dataset_A1, v=v, change_noise=True, param=param)
                if flag == False:
                    while flag==False:
                        mean_vel, cot, flag = run_policy_identify(env=VPenv, ws_dataset=ws_dataset_A1, v=v, change_noise=True, change_noise2=True, param=param)
            actual_velocity = np.append(actual_velocity,mean_vel)
            measured_cot = np.append(measured_cot, cot)
        print('')
        plot_data[param]={'vel':actual_velocity, 'cot':measured_cot}
        with open(os.path.join(os.getcwd(),'identify31.pickle'), mode='wb') as f:
            pickle.dump(plot_data, f)