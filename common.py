import os
import time
import random
import math
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from multiprocessing import Pool
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support

D = 10000
features = 81
threshhold = 0.2
sample_size = 10
eta_center = 0.01
eta_region = 0.001

L_spike = np.random.choice([-1, 1], size=D)
L_not_spike = (-1) * L_spike


class Encoder(object):

    def __init__(self, features : int, dim : int = 4000):
        self.dim = dim
        self.features = features
        self.basis = torch.randn(self.dim, self.features)
        self.base = torch.empty(self.dim).uniform_(0.0, 2*math.pi)

    def __call__(self, x : torch.Tensor):


        n = x.size(0)
        bsize = math.ceil(0.01*n)
        h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)
        temp = torch.empty(bsize, self.dim, device=x.device, dtype=x.dtype)

        # we need batches to remove memory usage
        for i in range(0, n, bsize):
            torch.matmul(x[i:i+bsize], self.basis.T, out=temp)
            torch.add(temp, self.base, out=h[i:i+bsize])
#             h[i:i+bsize].tanh_()i
            h[i:i+bsize].cos_().mul_(temp.sin_())
        return h

    def to(self, *args):

        self.basis = self.basis.to(*args)
        self.base = self.base.to(*args)
        return self


def encode_decode_position(List_of_positions):
    encode_pos = {}
    decode_pos = {}
    counter = 0 
    for pos in List_of_positions:
        encode_pos[pos] = counter
        decode_pos[counter] = pos
        counter +=1
    return encode_pos , decode_pos


def squeeze_input(encoded_stims):
    x = encoded_stims.values()
    x = np.array(list(x))
    x = np.squeeze(x)
    return x


def eucli_dis(Y_pos,Y_pred_pos):
    euc_dis = []
    for i in range(len(Y_pos)):
        euc_dis.append(euclidean_distance(Y_pos[i],Y_pred_pos[i]))
    return euc_dis


def Threshhoding_receptive_field(RF,threshhold):
    normalized_RF = RF/np.max(RF)
    normalized_RF[normalized_RF < threshhold] =0
    return normalized_RF


def interpolate_time_hypervectors_window(start_vec,end_vec,idx,D,window):
    interpolated_vec = np.zeros(D)
    interpolated_vec = (idx/window)*end_vec + (1-(idx/window))*start_vec
    return interpolated_vec


def ID_random_HyperVectors(num_of_HV,window):
    ID_random_HV = []
    ID_random_HV.append(0)
    num_random_HV = math.floor(num_of_HV/window)
    for i in range(num_random_HV):
        ID_random_HV.append((i+1)*(window))
    return ID_random_HV


def generate_random_vecotor(time_hypervectors,ID_random_HV,D):
    for idx in ID_random_HV:
        time_hypervectors[idx] = np.random.randint(2, size=D)
    return time_hypervectors


def get_unitID_in_bin(start_time,end_time,info_unit_spike):
    condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < end_time) &( start_time< info_unit_spike['time_since_stimulus_presentation_onset']))
    Unit_IDs = info_unit_spike.loc[condition,'unit_id']
    counts = Unit_IDs.value_counts()
    repeated_unit_ids = np.repeat(counts.index, counts.values)
    # Unit_IDs dataframe contains unit_IDs of neurons which had spikes in this interval. 
    return Unit_IDs.values , counts.to_dict()


def illumunation_HV():
    L_spike = np.random.choice([-1, 1], size=D)
    L_not_spike = (-1)*L_spike
    return L_spike , L_not_spike


def make_HV_of_units_in_bin(encoded_all_units,Unit_IDs):
    sum_encoded_units = np.zeros(D).astype(np.float64).reshape((1,D))
    for unit_id in Unit_IDs:
        sum_encoded_units += encoded_all_units[unit_id]
    return sum_encoded_units


def make_HV_of_units_in_binـillumination_method(encoded_all_units,Unit_IDs_in_bin):
    sum_encoded_units = np.zeros(D).astype(np.float64).reshape((1,D))
#     L_spike , L_not_spike
    for unit_id in encoded_all_units.keys():
        if unit_id in Unit_IDs_in_bin:
            sum_encoded_units +=  np.multiply(encoded_all_units[unit_id].reshape((1,D)),L_spike.reshape((1,D)))
        else :
            sum_encoded_units +=  np.multiply(encoded_all_units[unit_id].reshape((1,D)),L_not_spike.reshape((1,D)))
    return sum_encoded_units


def give_max_position(deltas):
    max_similarity = np.max(list(deltas.values()))
    for k, v in deltas.items():
        if v == max_similarity:
            return k


def give_max_position_randomly(deltas):
    max_similarity = np.max(list(deltas.values()))
#     print("deltas : ",deltas)
#     print("max_similarity : ",max_similarity)
    max_positions = []
    for k, v in deltas.items():
        if v == max_similarity:
            max_positions.append(k)
#     print("max_positions : ",max_positions)
    random_element = random.choice(max_positions)
#     print("random_element :",random_element)
    return random_element


def give_norm_each_class(Classes):
    for cls in Classes.keys():
        print(" cls : ",cls," norm : ",LA.norm(Classes[cls]))


def make_classes(List_of_positions):
    Classes = {}
    for c in List_of_positions:
        Classes[c] = np.zeros((1,D))
    return Classes


def check_similarity_to_all_classes(Classes,encoded_data_stim):
    deltas = {}
    for pos in Classes.keys():
        deltas[pos] = check_similarity(Classes[pos],encoded_data_stim)
    return deltas


def check_similarity(cls,encoded_data_stim):
#     print("encoded_data_stim",encoded_data_stim)
#     print("cls*encoded_data_stim",np.dot(cls,encoded_data_stim.reshape((D,1))))
#     print("LA.norm(cls)",LA.norm(cls))
#     print("LA.norm(encoded_data_stim)",LA.norm(encoded_data_stim))
#     print("cls.shape : ",cls.shape,"encoded_data_stim.shape : ",encoded_data_stim.shape )
    if LA.norm(cls) == 0 or LA.norm(encoded_data_stim) == 0  :
        return 0 
    else:
        similarity = np.dot(cls,encoded_data_stim.reshape((D,1)))/(LA.norm(cls)*LA.norm(encoded_data_stim))
#     print("similarity",similarity)
    return similarity


def check_in_region(pos,predicted_class):
#     print('pos',pos,'predicted_class',predicted_class)
    if np.abs(pos[0]-predicted_class[0])<2 and np.abs(pos[1]-predicted_class[1])<2:
        return True
    else:
        return False


def predicted_stim_in_different_position(Classes,List_of_positions,test_encoded_stims, grouped_by_Position,pos ):
    predicted_pos = make_accuracy_dict(List_of_positions)
    list_of_stim_same_pos = list(grouped_by_Position.get_group(pos))
    print(len(list_of_stim_same_pos))
    for stim in list_of_stim_same_pos:
        if stim in test_encoded_stims.keys():
            encoded_test_stim = test_encoded_stims[stim]
            deltas = check_similarity_to_all_classes(Classes,encoded_test_stim)
            predicted_class=max(deltas, key = deltas.get)
            predicted_pos[predicted_class] += (1.0/10)*100
    predicted_pos_df = plot_heatmap_of_region_accuracy(predicted_pos)
    return predicted_pos , predicted_pos_df


def make_accuracy_dict(List_of_positions):
    accuracy = {}
    for c in List_of_positions:
        accuracy[c] = 0
    return accuracy


def update_classes(Classes,deltas,encoded_data_stim,pos,deltas_pos,predicted_class,deltas_predicted_class,in_region):
    if in_region:
        Classes = update_correct_classes_in_region(Classes,encoded_data_stim,pos,deltas_pos)
    else:
        Classes = update_correct_classes(Classes,deltas,encoded_data_stim,pos,deltas_pos,deltas_predicted_class)
        Classes = update_incorrect_classes(Classes,deltas,encoded_data_stim,deltas_pos,predicted_class,deltas_predicted_class)
#     give_norm_each_class(Classes)
    return Classes


def update_correct_classes_in_region(Classes,encoded_data_stim,pos,deltas_pos):
    Classes[pos] += eta_center*(1 - deltas_pos)*encoded_data_stim
#     Classes[pos] += eta_center*encoded_data_stim
    neighbors_position = make_neighbor_position(pos)
    for position in neighbors_position:
        Classes[position] += eta_region*(1 - deltas_pos)*encoded_data_stim
    return Classes


def update_correct_classes(Classes,deltas,encoded_data_stim,pos,deltas_pos,deltas_predicted_class):
    Classes[pos] += eta_center*(deltas_predicted_class - deltas_pos)*encoded_data_stim
#     Classes[pos] += eta_center*encoded_data_stim
    neighbors_position = make_neighbor_position(pos)
    for position in neighbors_position:
        Classes[position] += eta_region*(deltas_predicted_class - deltas_pos)*encoded_data_stim
#         Classes[position] += eta_region*(deltas[position] - deltas_pos)*encoded_data_stim
    return Classes


def update_incorrect_classes(Classes,deltas,encoded_data_stim,deltas_pos,predicted_class,deltas_predicted_class):
    Classes[predicted_class] -= eta_center*(deltas_predicted_class - deltas_pos)*encoded_data_stim
    neighbors_position = make_neighbor_position(predicted_class)
    for position in neighbors_position:
        Classes[position] -= eta_region*(deltas_predicted_class - deltas_pos)*encoded_data_stim
#         Classes[position] -= eta_region*(deltas[position] - deltas_pos)*encoded_data_stim
    return Classes


def make_neighbor_position(pos):
    neighbors_position = []
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if pos[0]+i >= 0 and pos[1]+j >= 0 and pos[0]+i < 9 and pos[1]+j < 9: 
                neighbors_position.append((pos[0]+i,pos[1]+j))
    neighbors_position.remove(pos)
    return neighbors_position


def plot_heatmap_of_region_accuracy(accuracy,path):
    my_index = list(np.arange(8,-1,-1))
    my_columns =  list(np.arange(0,9))
    my_index = pd.Index(my_index)
    my_columns = pd.Index(my_columns)
    acc_df = pd.DataFrame(index=my_index, columns=my_columns)
    for pos in accuracy.keys():
        acc_df.loc[pos[0]][pos[1]] = accuracy[pos]
    acc_df = acc_df.astype(int)
    sns.heatmap(acc_df, cmap='coolwarm',annot=True)
    plt.savefig(os.path.join(path,'heatmap_plot.png'))
    plt.show()
#     acc_df = acc_df.pivot(index=my_index, columns=my_columns, values=acc_df)
#     sns.heatmap(acc_df, cmap='coolwarm')
#     plt.show()
    return


def euclidean_distance(x,y):
    return np.sqrt(((x[0] - y[0])**2)+((x[1] - y[1])**2))


def shuffle_nueros_spikes_in_stims(all_data):
    for i in range(all_data.shape[1]):
        np.random.shuffle(all_data[:,i,:])
    return all_data


def spikes_bins_table(info_unit_spike,time_bins,unit_IDs):
    
    spikes_table = pd.DataFrame(0,columns=time_bins[:-1], index=unit_IDs)
    for t in range(len(time_bins)-1):
        
        Unit_IDs_in_bin , Unit_IDs_counts = get_unitID_in_bin(time_bins[t],time_bins[t+1],info_unit_spike)
        for u, count in Unit_IDs_counts.items(): 
            spikes_table.loc[u, time_bins[t]] = count
        
    return spikes_table


def spikes_bins_table_overlapping(info_unit_spike,time_bins,unit_IDs):

    spikes_table = pd.DataFrame(0,columns=time_bins[:,0], index=unit_IDs)
    for n in range(len(time_bins)):
        
        Unit_IDs_in_bin , Unit_IDs_counts = get_unitID_in_bin(time_bins[n][0],time_bins[n][1],info_unit_spike)
        for u, count in Unit_IDs_counts.items(): 
            spikes_table.loc[u, time_bins[n][0]] = count
        
    return spikes_table


def encode_brain_regions(brain_regions):
    encoded_brain_regions = []
    for br in brain_regions:
        encoded_brain_regions.append(np.random.randint(2, size=D))
    return  encoded_brain_regions


def encode_brain_regions_dict(brain_regions):
    encoded_brain_regions = {}
    for br in brain_regions:
        encoded_brain_regions[br] = np.random.randint(2, size=D)
    return  encoded_brain_regions


def generate_time_slice_for_overlapping_bin(bin_):
    num_of_HV = int(np.ceil(0.25/bin_))
    time_bins = np.zeros((num_of_HV,2))
    each_step = (0.250-bin_)/int(0.250/bin_)
    for step in range(num_of_HV):
        time_bins[step][0] = step*each_step
        time_bins[step][1] = step*each_step + bin_
    return time_bins


def generate_time_slice_for_cropping(bin_):
    num_of_HV = int(np.floor(0.25/bin_))
    time_bins = np.zeros((num_of_HV,2))
    each_step = (0.250-bin_)/int(0.250/bin_)
    for step in range(num_of_HV):
        time_bins[step][0] = step*each_step
        time_bins[step][1] = step*each_step + bin_
    return time_bins


def Multiprocessing(func_name,input_):
    tic = time.time()


    with Pool(32) as p:

        Train_encoded_data = []
        Test_encoded_data = []
        ML_inputs_train = []
        ML_outputs_train = []
        ML_inputs_test = []
        ML_outputs_test = []

        results = p.imap(func_name, input_)
        for  train_encoded_data , test_encoded_data , ml_inputs_train , ml_outputs_train , ml_inputs_test , ml_outputs_test in results:

            Train_encoded_data.extend(train_encoded_data) 
            Test_encoded_data.extend(test_encoded_data)
            ML_inputs_train.extend(ml_inputs_train)
            ML_outputs_train.extend(ml_outputs_train)
            ML_inputs_test.extend(ml_inputs_test)
            ML_outputs_test.extend(ml_outputs_test)
    tac = time.time()

    return Train_encoded_data , Test_encoded_data , ML_inputs_train , ML_outputs_train , ML_inputs_test , ML_outputs_test


def save_inputs_to_directory(input0,  directory,str_bin):
    # Create the directory
    full_path = os.path.join(directory,str_bin)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    # input1, input2, input3, input4, input5, input6,
    
    # Save inputs separately in the directory
    np.save(os.path.join(full_path, 'result.npy'), np.array(input0))


def save_result_to_directory(result, name_to_save,  directory,str_bin):
    # Create the directory
    full_path = os.path.join(directory,str_bin)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    np.save(os.path.join(full_path, name_to_save+'.npy'), np.array(result))


def save_all_inputs_to_directory(input1,input2,input3,input4,input5,input6,  directory,str_bin):
    # Create the directory
    full_path = os.path.join(directory,str_bin)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    # input1, input2, input3, input4, input5, input6,
    
    # Save inputs separately in the directory
    # np.save(os.path.join(full_path, 'result.npy'), np.array(input0))
    np.save(os.path.join(full_path, 'Train_encoded_data.npy'), np.array(input1))
    np.save(os.path.join(full_path, 'Test_encoded_data.npy'), np.array(input2))
    np.save(os.path.join(full_path, 'ML_inputs_train.npy'), np.array(input3))
    np.save(os.path.join(full_path, 'ML_outputs_train.npy'), np.array(input4))
    np.save(os.path.join(full_path, 'ML_inputs_test.npy'), np.array(input5))
    np.save(os.path.join(full_path, 'ML_outputs_test.npy'), np.array(input6))


def save_all_inputs_to_directory_natural_scenes(input1,input2,input3,input4,input5,input6,input7,  directory,str_bin):
    # Create the directory
    full_path = os.path.join(directory,str_bin)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    # input1, input2, input3, input4, input5, input6,
    
    # Save inputs separately in the directory
    # np.save(os.path.join(full_path, 'result.npy'), np.array(input0))
    np.save(os.path.join(full_path, 'Train_encoded_data.npy'), np.array(input1))
    np.save(os.path.join(full_path, 'Test_encoded_data.npy'), np.array(input2))
    np.save(os.path.join(full_path, 'ML_inputs_train.npy'), np.array(input3))
    np.save(os.path.join(full_path, 'ML_outputs_train.npy'), np.array(input4))
    np.save(os.path.join(full_path, 'ML_inputs_test.npy'), np.array(input5))
    np.save(os.path.join(full_path, 'ML_outputs_test.npy'), np.array(input6))
    np.save(os.path.join(full_path, 'Test_To_Shuffle.npy'), np.array(input7))


def get_not_augmented_samples_gabor(encoded_data,initial_samples):
    num_repetitions = 81
    result_list = []
    for i in range(num_repetitions):
        start_row = i * 11*initial_samples
        end_row = initial_samples + i*11*initial_samples
        result_list.append(encoded_data[start_row:end_row, :])
    result_array = np.concatenate(result_list, axis=0)
    return result_array


def encode_brain_regions_EX_IN(brain_regions):
    encoded_brain_regions = []
    for br in brain_regions:
        encoded_brain_regions.append(np.random.randint(2, size=D))
        encoded_brain_regions.append(np.random.randint(2, size=D))
    return  np.array(encoded_brain_regions)


def encode_brain_regions_EX_IN_dict(brain_regions):
    encoded_brain_regions = {}
    for br in brain_regions:
        brain_region_HV = np.random.rand(D)
        encoded_brain_regions[f'{br}_EX'] = np.random.randint(2, size=D)*brain_region_HV
        encoded_brain_regions[f'{br}_IN'] = np.random.randint(2, size=D)*brain_region_HV
    return  encoded_brain_regions


def get_num_of_spikes_in_bins_whole_brain(start_time,end_time,spikes_whole_brain,stim):
    
    info_unit_spike = spikes_whole_brain[spikes_whole_brain['stimulus_presentation_id'] == stim]
    condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < end_time) & (start_time < info_unit_spike['time_since_stimulus_presentation_onset']))
    df = info_unit_spike.loc[condition]
    return df.shape[0]


def averaging_metrics(y_true, y_pred,average):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    if average == 'micro':
        print(f"Micro-averaged precision: {precision}")
        print(f"Micro-averaged recall: {recall}")
        print(f"Micro-averaged F1-score: {f1}")
    elif average == 'macro':
        print(f"Macro-averaged precision: {precision}")
        print(f"Macro-averaged recall: {recall}")
        print(f"Macro-averaged F1-score: {f1}")
    return precision, recall, f1


def give_max_frame_randomly(deltas):
    max_similarity = np.max(list(deltas.values()))
#     print("deltas : ",deltas)
#     print("max_similarity : ",max_similarity)
    max_positions = []
    for k, v in deltas.items():
        if v == max_similarity:
            max_positions.append(k)
#     print("max_positions : ",max_positions)
    random_element = random.choice(max_positions)
#     print("random_element :",random_element)
    return random_element


def load_data(session_id,averaging_level, Train_encoded_data_or,Test_encoded_data_or,ML_inputs_train_or,ML_outputs_train_or,ML_inputs_test_or,ML_outputs_test_or):
    Train_encoded_data_or  = Train_encoded_data_or.replace('757970808', str(session_id))
    Test_encoded_data_or = Test_encoded_data_or.replace('757970808', str(session_id))
    ML_inputs_train_or = ML_inputs_train_or.replace('757970808', str(session_id))
    ML_outputs_train_or = ML_outputs_train_or.replace('757970808', str(session_id))
    ML_inputs_test_or = ML_inputs_test_or.replace('757970808', str(session_id))
    ML_outputs_test_or = ML_outputs_test_or.replace('757970808', str(session_id))
    
    Train_encoded_data_or  = Train_encoded_data_or.replace('neuron_level', averaging_level)
    Test_encoded_data_or = Test_encoded_data_or.replace('neuron_level', averaging_level)
    ML_inputs_train_or = ML_inputs_train_or.replace('neuron_level', averaging_level)
    ML_outputs_train_or = ML_outputs_train_or.replace('neuron_level', averaging_level)
    ML_inputs_test_or = ML_inputs_test_or.replace('neuron_level', averaging_level)
    ML_outputs_test_or = ML_outputs_test_or.replace('neuron_level', averaging_level)
    
    
    Train_encoded_data= np.load( Train_encoded_data_or,allow_pickle=True)
    Test_encoded_data = np.load( Test_encoded_data_or,allow_pickle=True)
    ML_inputs_train = np.load( ML_inputs_train_or,allow_pickle=True)
    ML_outputs_train = np.load( ML_outputs_train_or,allow_pickle=True)
    ML_inputs_test = np.load( ML_inputs_test_or,allow_pickle=True)
    ML_outputs_test= np.load( ML_outputs_test_or,allow_pickle=True)

    return Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test
