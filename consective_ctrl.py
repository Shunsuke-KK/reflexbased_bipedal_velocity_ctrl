import os
import pickle
import pandas as pd
from pandas import DataFrame
import numpy as np
import math
from custom_env import Reflex_WALK_Env
from reflex_opt.run_policy_consective import run_policy_consective

forder_name = 'opt_ex'
save_path = os.path.join(os.getcwd(),'reflex_opt/save_data')
save_folder = save_path + '/' + forder_name


def l_s_method_original(x, y, weight, degree):
    t = y
    t = t*weight
    phi = DataFrame()
    for i in range(0, degree+1):
        p = x ** i
        p = p*weight
        p.name = "x ** %d" % i
        phi = pd.concat([phi, p], axis = 1)
    ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi).astype(np.float64)), phi.T), t)
    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return (f, ws)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    path=os.getcwd()+'/assets'
    VPenv = Reflex_WALK_Env(path=path)
    pickle_data = True
    pickle_data = False

    if not pickle_data:
        params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA',
                'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
                'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v',
                'vel','cot','value']
        data = {}
        optimizer = {}
        dataset = DataFrame(columns = params_label)
        num = 0
        while True:
            filename = f'vel_gen{num}.pickle'
            if os.path.exists(os.path.join(save_folder,filename)):
                with open(os.path.join(save_folder,filename), mode='rb') as f:
                    load_checkpoint = pickle.load(f)
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
        #1 #800 #100000
        coefficient = 1 #1 #800 #5000
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


    if pickle_data:
        with open('functions/function_A800_27.pickle', mode='rb') as g:
            load_checkpoint = pickle.load(g)
            ws_dataset = load_checkpoint['ws_dataset']

    run_policy_consective(env=VPenv, ws_dataset=ws_dataset)