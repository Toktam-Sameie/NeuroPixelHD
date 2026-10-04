import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import unravel_index
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping
import math
import torch
from sklearn.neighbors import KernelDensity
from tqdm.auto import tqdm
import ast
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from common import (
    Encoder,
    Multiprocessing,
    Threshhoding_receptive_field,
    averaging_metrics,
    check_in_region,
    check_similarity_to_all_classes,
    encode_brain_regions_EX_IN_dict,
    eucli_dis,
    euclidean_distance,
    generate_random_vecotor,
    get_num_of_spikes_in_bins_whole_brain,
    get_unitID_in_bin,
    give_max_frame_randomly,
    give_max_position_randomly,
    interpolate_time_hypervectors_window,
    make_HV_of_units_in_binـillumination_method,
    make_accuracy_dict,
    make_classes,
    plot_heatmap_of_region_accuracy,
    save_inputs_to_directory,
    shuffle_nueros_spikes_in_stims,
    spikes_bins_table,
    update_classes,
)
output_dir = 'ecephys_cache_dir/'
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions_wild = brain_observatory_type_sessions[brain_observatory_type_sessions["full_genotype"] == 'wt/wt']

list_of_sessions_wild_type = list(brain_observatory_type_sessions_wild.index)


#         if not brain_region_units.empty:
#             br_regions.append(br)
#             print('brain_region_units',brain_region_units)
#     return br_regions

def whole_brain_regions_check(brain_regions):
    br_regions = []
    for br in brain_regions:
        if br not in eligible_brain_regions:
            continue
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]
        spikes = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )

        if not spikes.empty:
            br_regions.append(br)
            
    return br_regions
    
eligible_brain_regions = ['VISp','VISam', 'VISal', 'VISrl', 'VISmma','VISpm', 'VISl','CA1', 'CA2', 'CA3','SUB', 'ProS','DG','TH', 'LP', 'LGv', 'LGd', 'PP', 'PIL', 'MGv', 'PO', 'Eth', 'POL','APN','ZI']


#Parameters
D = 10000
features = 81
threshhold = 0.2
sample_size = 10
eta_center = 0.01
eta_region = 0.001

L_spike = np.random.choice([-1, 1], size=D)
L_not_spike = (-1)*L_spike


def target_pos_to_num(target_pos,encode_pos):
    Y_num = []
    for i in range(len(target_pos)):
        Y_num.append(encode_pos[target_pos[i]])
    return Y_num


def get_Y_pos_to_num(encoded_stims):    
    encoded_Y = []
    tuple_pos_Y =[]
    for stim in encoded_stims.keys():
        pos = rf_stim_table.iloc[stim-1]['Position']
        encoded_Y.append(encode_pos[pos])
        tuple_pos_Y.append(pos)
    return encoded_Y , tuple_pos_Y


def label_to_tuple_pos(y_pred):  
    y_tuple_pos = []
    for yp in list(y_pred):
        y_tuple_pos.append(decode_pos[yp])
    return y_tuple_pos


def Test_model_in_region_label_to_tuple_pos(y,y_pred,num_stim_group):
    accuarcy = make_accuracy_dict(List_of_positions)
    y_tuple_pos = []
    for i,yp in enumerate(y_pred):
        pred_pos = decode_pos[yp]
        y_tuple_pos.append(pred_pos)
        pos = decode_pos[y[i]]
        in_region = check_in_region(pos,pred_pos)
        if in_region:
            accuarcy[pos] += (1.0/num_stim_group)*100
            
    return accuarcy , y_tuple_pos


def Random_forest(brain_region,Train_input,Train_target,Test_input,Test_target):
    
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=200)

    rfc.fit(Train_input, list(Train_target))

    y_pred_train = rfc.predict(Train_input)
    y_pred_test = rfc.predict(Test_input)
    y_tuple_pos_train = label_to_tuple_pos(y_pred_train)
    y_tuple_pos_test = label_to_tuple_pos(y_pred_test)
    Test_target_pos = label_to_tuple_pos(Test_target)
    euc_dis_test_RF = eucli_dis(Test_target_pos,y_tuple_pos_test)
    return euc_dis_test_RF


def SVM(brain_region,Train_input,Train_target,Test_input,Test_target):
    
    from sklearn import svm
    svmc = svm.SVC(kernel='rbf', C=1, decision_function_shape='ovr')

    svmc.fit(Train_input, list(Train_target))
    print('done fitting with svm')
    y_pred_test = svmc.predict(Test_input)

    y_tuple_pos_test = label_to_tuple_pos(y_pred_test)
    Test_target_pos = label_to_tuple_pos(Test_target)
    euc_dis_test_svm = eucli_dis(Test_target_pos,y_tuple_pos_test)
    return euc_dis_test_svm


# we use this function to encode every unit based in the unit_id
def encode_receptive_field(unit_id,encoder,threshhold):
    RF = rf_mapping.get_receptive_field(unit_id)
    if np.max(RF) == 0:
        return np.random.rand(D)
    RF = Threshhoding_receptive_field(RF,threshhold)
    flatten_RF = RF.flatten()
    tensor_RF = torch.tensor(flatten_RF).type('torch.FloatTensor')
    encoded_RF = encoder(tensor_RF.reshape((1,81)))
    encoded_RF= encoded_RF.numpy()
    return encoded_RF

def generate_time_hypervector(bin_,D):
    
    num_of_HV = int(0.25/bin_)
    ID_random_HV = []
    ID_random_HV.append(0)
    ID_random_HV.append(num_of_HV-1)
    window_list = np.arange(ID_random_HV[0]+1,ID_random_HV[1])
    #generate random vecotor for each of index in ID_random_HV
    time_hypervectors = np.zeros((num_of_HV,D))
    time_hypervectors = generate_random_vecotor(time_hypervectors,ID_random_HV,D)
    
    for i in window_list:
        time_hypervectors[i] = interpolate_time_hypervectors_window(time_hypervectors[ID_random_HV[0]],time_hypervectors[ID_random_HV[1]],i,D,num_of_HV-1)
    return time_hypervectors


def spatial_temporal_encoding_each_stimulation(spikes,encoded_all_units,stim,bin_):
    info_unit_spike = spikes[spikes['stimulus_presentation_id']==stim]
    spatial_temporal_encoded_all_bins = np.zeros((1,D))
    #we have to slice during of each stimulation to 0.01 s and see which units spike in this interval
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    encoded_time_hypervector = generate_time_hypervector(bin_,D)
    for t in range(len(time_bins)-1):
        Unit_IDs_in_bin , Unit_IDs_count = get_unitID_in_bin(time_bins[t],time_bins[t+1],info_unit_spike)
        sum_encoded_units = make_HV_of_units_in_binـillumination_method(encoded_all_units,Unit_IDs_in_bin)
        spatial_temporal_encoded_all_bins += encoded_time_hypervector[t] * sum_encoded_units
    return spatial_temporal_encoded_all_bins


def Test_model_in_region_accuracy(Classes,dataset,num_of_sample_each_pos):
    accuarcy = make_accuracy_dict(List_of_positions)
    euclidean_dis = []
    for i in range(len(dataset)):
        pos = dataset[i][0]
        data = dataset[i][1]
        
        
        deltas = check_similarity_to_all_classes(Classes,data)
        predicted_class=max(deltas, key = deltas.get)
        euclidean_dis.append(euclidean_distance(pos,predicted_class))
        in_region = check_in_region(pos,predicted_class)
        if in_region:
            accuarcy[pos] += (1.0/num_of_sample_each_pos)*100
    return accuarcy , euclidean_dis 

def Train_Test_model(brain_region,epochs,shuffle):
    Classes , train_encoded_stims , test_encoded_stims =  Adaptive_training_with_test(brain_region,List_of_positions,epochs,shuffle)
    accuarcy_test,euclidean_dis_test = Test_model_in_region_accuracy(Classes,test_encoded_stims,10)
    accuarcy_train , euclidean_dis_train = Test_model_in_region_accuracy(Classes,train_encoded_stims,35)
    accuarcy_train_df = plot_heatmap_of_region_accuracy(accuarcy_train)
    accuarcy_test_df = plot_heatmap_of_region_accuracy(accuarcy_test)
    return accuarcy_train ,accuarcy_test , euclidean_dis_train , euclidean_dis_test 


from scipy.stats import wilcoxon
def Wilcoxon_signed_rank(X,Y,alternative):
    res = wilcoxon(X,Y,alternative=alternative)
    print(" pvalue : ",res.pvalue)
    return res.pvalue


def spikes_seperate_brain_region(brain_regions):
    br_spikes = {}
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]

        spikes_br = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        br_spikes[br] = spikes_br 
    return br_spikes

def spikes_diff_brain_region(brain_regions):
    spikes = pd.DataFrame()
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]

        spikes_br = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        if not spikes_br.empty:
            spikes = pd.concat([spikes, spikes_br], axis=0)

    return spikes


def encode_all_units_seperating_region(brain_regions):
    encoded_all_units = []
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]
        brain_region_HV = np.random.rand(D)
        spikes_br = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        unit_IDs_br = list(spikes_br.unit_id.unique())
        
        for unitID in unit_IDs_br:
            encoded_all_units.append(brain_region_HV * encode_receptive_field(unitID,encoder,threshhold))
    return np.squeeze(np.array(encoded_all_units), axis=1)
    
def encode_all_units_seperating_region_dict(brain_regions):
    encoded_all_units = {}
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]
        brain_region_HV = np.random.rand(D)
        spikes_br = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        unit_IDs_br = list(spikes_br.unit_id.unique())
        
        for unitID in unit_IDs_br:
            encoded_all_units[unitID]=brain_region_HV * encode_receptive_field(unitID,encoder,threshhold)
    return encoded_all_units


def data_augmentation_shuffling_method_neuron_level(pos):
  

    Train_encoded_data = []
    Test_encoded_data = []
    ML_inputs_train = []
    ML_outputs_train = []
    ML_inputs_test = []
    ML_outputs_test = []
    
    tic_pos =  time.time()

    list_of_stim_same_pos = list(grouped_by_Position.get_group(pos))
    train_data_each_pos_to_shuffle = [] #np.zeros((len(list_of_stim_same_pos),1))
    test_data_each_pos_to_shuffle = []

    random_index_test = random.sample(list_of_stim_same_pos, sample_size)
    random_index_train = [x for x in list_of_stim_same_pos if x not in random_index_test]

    for stim in list_of_stim_same_pos:
        # Now I should assign 1 to units which in a specific time bin have spiked 
        

        info_unit_spike = spikes[spikes['stimulus_presentation_id']==stim]
        spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
        vectorized_spikes_table = np.hstack(spikes_table.to_numpy())

        if stim in random_index_test:
            ML_inputs_test.append(vectorized_spikes_table)
            ML_outputs_test.append(pos)
            test_data_each_pos_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe_new(spikes_table,encoded_all_units)
            Test_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])

        else :
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(pos)

            train_data_each_pos_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe_new(spikes_table,encoded_all_units)

            Train_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])

    train_data_each_pos_to_shuffle = np.array(train_data_each_pos_to_shuffle)
    test_data_each_pos_to_shuffle = np.array(test_data_each_pos_to_shuffle)


    for shuffle_time in range(shuffle_times):
        tic = time.time()
        shuffled_train_data_each_pos = shuffle_nueros_spikes_in_stims(train_data_each_pos_to_shuffle)

        shuffled_test_data_each_pos = shuffle_nueros_spikes_in_stims(test_data_each_pos_to_shuffle)


        list_of_generated_stim_train = list(np.arange(shuffled_train_data_each_pos.shape[0]))
        list_of_generated_stim_test = list(np.arange(shuffled_test_data_each_pos.shape[0]))

        for i in list_of_generated_stim_test:
            tic_encoding = time.time()
            reshaped_array = shuffled_test_data_each_pos[i]

            encoded_data_for_each_stim = encode_generated_dataframe_new(reshaped_array,encoded_all_units)
            vectorized_spikes_table = np.hstack(reshaped_array)
            Test_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])
            ML_inputs_test.append(vectorized_spikes_table)
            ML_outputs_test.append(pos)

        for i in list_of_generated_stim_train:
            reshaped_array = shuffled_train_data_each_pos[i]

            encoded_data_for_each_stim = encode_generated_dataframe_new(reshaped_array,encoded_all_units)
            vectorized_spikes_table = np.hstack(reshaped_array)
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(pos)

            Train_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])

    return Train_encoded_data , Test_encoded_data , ML_inputs_train , ML_outputs_train , ML_inputs_test , ML_outputs_test


def encode_generated_dataframe_new(spikes_table,encoded_all_units):
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    encoded_time_hypervector = generate_time_hypervector(bin_,D)
    #step 1 
    # do something with encoded_all_units,L_spike
    neuron_spiked = encoded_all_units*L_spike.reshape(1,D)
    sum_encoded_units_spiked = np.matmul(np.transpose(spikes_table), neuron_spiked)
    #step 2
    neuron_not_spiked = encoded_all_units*L_not_spike.reshape(1,D)
    spikes_table_logical_not = np.where(spikes_table == 0, 1, 0)
    sum_encoded_units_not_spiked = np.matmul(np.transpose(spikes_table_logical_not), neuron_not_spiked)
    
    sum_encoded_units = sum_encoded_units_not_spiked+ sum_encoded_units_spiked
    spatial_temporal_encoded_all_bins = np.sum(sum_encoded_units * encoded_time_hypervector, axis=0)
    print('spatial_temporal_encoded_all_bins',spatial_temporal_encoded_all_bins)
    return spatial_temporal_encoded_all_bins 


def Adaptive_training_data_augmentation_shuffling(Train_data,Test_data,brain_region,List_of_positions,epochs,suffle_times):
    Train_sample_each_pos=(suffle_times+1)*(45 - sample_size)
    Test_sample_each_pos = (suffle_times+1)*sample_size
    
    random.shuffle(Train_data)
    random.shuffle(Test_data)
    Classes = make_classes(List_of_positions)
    # training model
    for epoch in tqdm(range(epochs), ascii=True,desc='training on epochs'):
        for i in range(len(Train_data)):

            pos = Train_data[i][0]
            data = Train_data[i][1].astype(np.float64)
            deltas = check_similarity_to_all_classes(Classes,data)
            predicted_class = give_max_position_randomly(deltas)
            in_region = check_in_region(pos,predicted_class)
            Classes = update_classes(Classes,deltas,data,pos,deltas[pos],predicted_class,deltas[predicted_class],in_region)

    
        print("________________epoch________________ :  ",epoch)
        print("accuarcy_train : ")
        print("accuarcy_test  : ")
    print('------------------final result------------------')
    accuarcy_test, euclidean_dis_test = Test_model_in_region_accuracy(Classes,Test_data,Test_sample_each_pos)
    return euclidean_dis_test , accuarcy_test


def augmenting_data_plus_training_temporal_averaging(bin_,brain_regions,epochs,shuffle_times):

    Train_encoded_data , Test_encoded_data , ML_inputs_train , ML_outputs_train , ML_inputs_test ,\
    ML_outputs_test = Multiprocessing(data_augmentation_shuffling_method_neuron_level,List_of_positions)
    
    euclidean_dis , accuarcy_test  =  Adaptive_training_data_augmentation_shuffling(Train_encoded_data,Test_encoded_data,brain_regions,List_of_positions,epochs,shuffle_times)
    
    save_inputs_to_directory(euclidean_dis,gabor_temporal_averaging_directory,str(bin_))
    return accuarcy_test 


#     #encode all units
    

 ##################################################################################  spatial averaging


def brain_region_EX_IN_unitsـFano_Factor_Gabor(brain_regions):
    print('brain_region_EX_IN_unitsـFano_Factor_Gabor')
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    
    br_EX_IN_units = {}
    
    List_of_positions = list(np.sort(rf_stim_table.Position.unique()))
    grouped_by_Position = rf_stim_table.groupby('Position')["stimulus_presentation_id"]
    
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]
        spikes = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        if spikes.empty:
            continue
                
        unit_IDs = list(spikes.unit_id.unique())
    
        
        all_spikes_tables_for_all_pos = pd.DataFrame()
        for pos in  List_of_positions:

            list_of_stim_same_pos = list(grouped_by_Position.get_group(pos))
            for stim in list_of_stim_same_pos:
                info_unit_spike = spikes[spikes['stimulus_presentation_id']==stim]
                spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
                all_spikes_tables_for_one_pos = pd.concat([all_spikes_tables_for_all_pos, spikes_table], axis=1)
        df = all_spikes_tables_for_one_pos
        row_means = df.mean(axis=1)

        # Calculate variance of each row (unit)
        row_variances = df.var(axis=1)

        # Divide variance by mean for each row (unit)
        fano_factors = np.where(row_means != 0, row_variances / row_means, 0)   
        sorted_fano_factors = np.sort(fano_factors)
        threshhold = sorted_fano_factors[int(len(sorted_fano_factors) * 0.8)]

        label_assignment = np.where(fano_factors >= threshhold , 'inhibitory', 'excitatory')
        br_EX_IN_units[br] = dict(zip(unit_IDs, label_assignment))
        print(br_EX_IN_units[br])
    return br_EX_IN_units 

def brain_region_EX_IN_unitsـFano_Factor_Natural_scenes(brain_regions):
    print('brain_region_EX_IN_unitsـFano_Facto_Natural_scenes')
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    
    br_EX_IN_units = {}
    
    List_of_positions = list(np.sort(rf_stim_table.frame.unique()))
    grouped_by_Position = rf_stim_table.groupby('frame')["stimulus_presentation_id"]
    
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]
        spikes = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        if spikes.empty:
            continue
                
        unit_IDs = list(spikes.unit_id.unique())
    
        
        all_spikes_tables_for_all_pos = pd.DataFrame()
        for pos in  List_of_positions:

            list_of_stim_same_pos = list(grouped_by_Position.get_group(pos))
            for stim in list_of_stim_same_pos:
                info_unit_spike = spikes[spikes['stimulus_presentation_id']==stim]
                spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
                all_spikes_tables_for_one_pos = pd.concat([all_spikes_tables_for_all_pos, spikes_table], axis=1)
        df = all_spikes_tables_for_one_pos
        row_means = df.mean(axis=1)

        # Calculate variance of each row (unit)
        row_variances = df.var(axis=1)

        # Divide variance by mean for each row (unit)
        fano_factors = np.where(row_means != 0, row_variances / row_means, 0)   
        sorted_fano_factors = np.sort(fano_factors)
        threshhold = sorted_fano_factors[int(len(sorted_fano_factors) * 0.8)]

        label_assignment = np.where(fano_factors >= threshhold , 'inhibitory', 'excitatory')
        br_EX_IN_units[br] = dict(zip(unit_IDs, label_assignment))
        print(br_EX_IN_units[br])
    return br_EX_IN_units 


def spikes_bins_table_spatial_averaging_EX_IN(brain_regions,br_EX_IN_units,time_bins,stim):

      
    #then I have to build a dataframe of timebins versus neurons
    spikes_table = pd.DataFrame(columns=time_bins[:-1])

    # Iterate over brain regions
    for region in brain_regions:
        print(region)
        ex_row = pd.DataFrame(index=[f'{region}_EX'], columns=time_bins[:-1])
        in_row = pd.DataFrame(index=[f'{region}_IN'], columns=time_bins[:-1])
        
        # find the number of excitatory and inhibitory neurons in each brain region
        EX_IN_units = br_EX_IN_units[region]
        spikes_br = br_spikes[region]
        info_unit_spike = spikes_br[spikes_br['stimulus_presentation_id'] == stim]
        for t in range(len(time_bins)-1):
            condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < time_bins[t+1]) &( time_bins[t]< info_unit_spike['time_since_stimulus_presentation_onset']))
            Unit_IDs = info_unit_spike.loc[condition,'unit_id']
            unique_units_in_bin = list(Unit_IDs.unique())
            counts = Unit_IDs.value_counts()
            sum_ex = 0 
            sum_in = 0

            for key, value in counts.items():

                if EX_IN_units[key] == 'excitatory':
                    sum_ex += value

                elif EX_IN_units[key] == 'inhibitory':
                    sum_in += value

        
        # Create row for excitatory category
        # Assign values to the row
#             print('sum_ex',sum_ex)
            ex_row.loc[f'{region}_EX'][t] = sum_ex  

        # Create row for inhibitory category
        # Assign values to the row
#             print('sum_in ',sum_in )
            in_row.loc[f'{region}_IN'][t] = sum_in  

        # Append the rows to the dataframe
        
        spikes_table = spikes_table.append(ex_row)
        spikes_table = spikes_table.append(in_row) 
        
    return spikes_table

    
def get_unitID_in_bin_EX_IN(start_time,end_time,brain_regions):
    condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < end_time) &( start_time< info_unit_spike['time_since_stimulus_presentation_onset']))
    Unit_IDs = info_unit_spike.loc[condition,'unit_id']
    
    counts = Unit_IDs.value_counts()
    repeated_unit_ids = np.repeat(counts.index, counts.values)
    # Unit_IDs dataframe contains unit_IDs of neurons which had spikes in this interval. 
#     return Unit_IDs.values
    return list(repeated_unit_ids) , counts.to_dict()


def get_num_of_spikes_in_bins(start_time,end_time,brain_regions,stim):
    num_of_spikes_each_br = {}
    for br in brain_regions:

        spikes_br = br_spikes[br] 

        info_unit_spike = spikes_br[spikes_br['stimulus_presentation_id'] == stim]
        condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < end_time) & (start_time < info_unit_spike['time_since_stimulus_presentation_onset']))
        df = info_unit_spike.loc[condition]
        num_of_spikes_each_br[br] = df.shape[0]
    return num_of_spikes_each_br


def encode_generated_dataframe_EX_IN(spikes_table,encoded_brain_regions_EX_IN):
        
    spatial_temporal_encoded_all_bins = np.zeros((1,D))
    #we have to slice during of each stimulation to 0.01 s and see which units spike in this interval
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    encoded_time_hypervector = generate_time_hypervector(bin_,D)
    for t in range(len(time_bins)-1):

        neuron_types_in_bin = []
        for index, value in spikes_table[time_bins[t]].iteritems():
            if value != 0:
                neuron_types_in_bin.extend([index] * value)
        neuron_types_in_bin = np.array(neuron_types_in_bin)

        sum_encoded_units = make_HV_of_units_in_binـillumination_method(encoded_brain_regions_EX_IN,neuron_types_in_bin)
        spatial_temporal_encoded_all_bins += encoded_time_hypervector[t] * sum_encoded_units
    return spatial_temporal_encoded_all_bins


def spikes_bins_table_spatial_averaging(brain_regions,time_bins,stim):

    #then I have to build a dataframe of timebins versus neurons
    spikes_table = pd.DataFrame(0,columns=time_bins[:-1], index=brain_regions)
    
    # Now I should assign 1 to units which in a specific time bin have spiked  
    
    for t in range(len(time_bins)-1):
        br_spikes_number = get_num_of_spikes_in_bins(time_bins[t],time_bins[t+1],brain_regions,stim)
        for br, count in br_spikes_number.items(): 
            spikes_table.loc[br, time_bins[t]] = count
        
    return spikes_table


def get_num_of_spikes_in_bins(start_time,end_time,brain_regions,stim):
    num_of_spikes_each_br = {}
    for br in brain_regions:

        spikes_br = br_spikes[br] 
        info_unit_spike = spikes_br[spikes_br['stimulus_presentation_id'] == stim]
        condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < end_time) & (start_time < info_unit_spike['time_since_stimulus_presentation_onset']))
        df = info_unit_spike.loc[condition]
        num_of_spikes_each_br[br] = df.shape[0]
    return num_of_spikes_each_br

def spikes_seperate_brain_region(brain_regions):
    br_spikes = {}
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]

        spikes_br = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        br_spikes[br] = spikes_br 
    return br_spikes

def encode_all_units_seperating_region(brain_regions):
    encoded_all_units = []
    for br in brain_regions:
        brain_region_units = session.units[session.units["ecephys_structure_acronym"] == br]
        brain_region_HV = np.random.rand(D)
        spikes_br = session.presentationwise_spike_times(
            stimulus_presentation_ids=scene_presentations.index.values,
            unit_ids=brain_region_units.index.values[:]
        )
        unit_IDs_br = list(spikes_br.unit_id.unique())
        
        for unitID in unit_IDs_br:
            encoded_all_units.append(brain_region_HV * encode_receptive_field(unitID,encoder,threshhold))
    return np.squeeze(np.array(encoded_all_units), axis=1)


def processing_and_augmenting_train_data_gabor(args): 
    pos,spatial_averaging_level = args
    shuffle_times_train = 1
    
    Test_encoded_data = []
    Train_encoded_data = []
    ML_inputs_train = []
    ML_outputs_train = []
    ML_inputs_test = []
    ML_outputs_test = []

    
    list_of_stim_same_pos = list(grouped_by_Position.get_group(pos))
    train_data_each_pos_to_shuffle = [] #np.zeros((len(list_of_stim_same_pos),1))
    test_data_each_pos_to_shuffle = []

    random_index_test = random.sample(list_of_stim_same_pos, sample_size)
    random_index_train = [x for x in list_of_stim_same_pos if x not in random_index_test]

    for stim in list_of_stim_same_pos:
#         print('-----stim-----',stim)
#         tic = time.time()
        # Now I should assign 1 to units which in a specific time bin have spiked 
        if spatial_averaging_level == 'neuron_level':
            info_unit_spike = spikes[spikes['stimulus_presentation_id']==stim]
            spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
            
        
        elif spatial_averaging_level == 'population_level':
            spikes_table = spikes_bins_table_spatial_averaging_EX_IN(brain_regions,br_EX_IN_units,time_bins,stim)
            
        elif spatial_averaging_level == 'each_region_level':
            spikes_table = spikes_bins_table_spatial_averaging(brain_regions,time_bins,stim)
            
            
        elif spatial_averaging_level == 'super_region_level':
            spikes_table = spikes_bins_table_spatial_averaging(brain_regions,time_bins,stim)
            
            
        elif spatial_averaging_level == 'whole_brain':
            spikes_table = spikes_bins_table_spatial_averaging_whole_brain(spikes_whole_brain,time_bins,stim)
            

        vectorized_spikes_table = np.hstack(spikes_table.to_numpy())

        if stim in random_index_test:
            ML_inputs_test.append(vectorized_spikes_table)
            ML_outputs_test.append(pos)
            test_data_each_pos_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe(spikes_table,encoded_all_units)

            Test_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])
        else :
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(pos)

            train_data_each_pos_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe(spikes_table,encoded_all_units)

            Train_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])
            
#         tac = time.time()
#         print( '______time __________',tac - tic)
#         return 
#     return Train_encoded_data , Test_encoded_data
    train_data_each_pos_to_shuffle = np.array(train_data_each_pos_to_shuffle)
    test_data_each_pos_to_shuffle = np.array(test_data_each_pos_to_shuffle)

#     Test_To_Shuffle.append(test_data_each_pos_to_shuffle )
    # Shuffling the Train data

    for shuffle_time in range(shuffle_times):

        shuffled_train_data_each_pos = shuffle_nueros_spikes_in_stims(train_data_each_pos_to_shuffle)
        shuffled_test_data_each_pos = shuffle_nueros_spikes_in_stims(test_data_each_pos_to_shuffle)

        list_of_generated_stim_train = list(np.arange(shuffled_train_data_each_pos.shape[0]))
        list_of_generated_stim_test = list(np.arange(shuffled_test_data_each_pos.shape[0]))

        for i in list_of_generated_stim_train:
            reshaped_array = shuffled_train_data_each_pos[i]
            generated_spikes_table = pd.DataFrame(reshaped_array,index=spikes_table.index, columns=spikes_table.columns)
            # Now I should do something with this dataframe (encoding it) (like spatioal_temporal_encoding function)
            encoded_data_for_each_stim = encode_generated_dataframe(generated_spikes_table,encoded_all_units)
            vectorized_spikes_table = np.hstack(reshaped_array)
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(pos)

            Train_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])


        for i in list_of_generated_stim_test:
            reshaped_array = shuffled_test_data_each_pos[i]
            generated_spikes_table = pd.DataFrame(reshaped_array,index=spikes_table.index, columns=spikes_table.columns)
            # Now I should do something with this dataframe (encoding it) (like spatioal_temporal_encoding function)
            encoded_data_for_each_stim = encode_generated_dataframe(generated_spikes_table,encoded_all_units)
            vectorized_spikes_table = np.hstack(reshaped_array)
            ML_inputs_test.append(vectorized_spikes_table)
            ML_outputs_test.append(pos)

            Test_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])
    return     Train_encoded_data ,Test_encoded_data  ,ML_inputs_train ,ML_outputs_train ,ML_inputs_test ,ML_outputs_test  


#def encode_generated_dataframe_new(spikes_table,encoded_all_units):
#     print('encoded_all_units',encoded_all_units.shape)
#     print('spikes_table',spikes_table.shape)
 #   time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
 #   encoded_time_hypervector = generate_time_hypervector(bin_,D)
  #  sum_encoded_units = np.matmul(np.transpose(spikes_table), encoded_all_units)
   # spatial_temporal_encoded_all_bins = np.sum(sum_encoded_units * encoded_time_hypervector, axis=0)
   # return spatial_temporal_encoded_all_bins 


def data_augmentation_shuffling_method(brain_regions,spatial_averaging_level):

    tic = time.time() 
    spatial_averaging_level_vector = np.repeat(spatial_averaging_level, 81)
    print("start preparing data")
    #_________________________________
    
    with Pool(32) as p:

     
        Test_encoded_data = []
        Train_encoded_data = []
        ML_inputs_train = []
        ML_outputs_train = []
        ML_inputs_test = []
        ML_outputs_test = []

        results = p.imap(processing_and_augmenting_train_data_gabor, zip(List_of_positions,spatial_averaging_level_vector))
        for train_encoded_data, test_encoded_data  ,ml_inputs_train ,ml_outputs_train ,ml_inputs_test ,ml_outputs_test in results:

            Train_encoded_data.extend(train_encoded_data) 
            Test_encoded_data.extend(test_encoded_data)
            ML_inputs_train.extend(ml_inputs_train)
            ML_outputs_train.extend(ml_outputs_train)
            ML_inputs_test.extend(ml_inputs_test)
            ML_outputs_test.extend(ml_outputs_test)
            
    return Train_encoded_data, Test_encoded_data , ML_inputs_train ,ML_outputs_train  , ML_inputs_test , ML_outputs_test


def spikes_bins_table_spatial_averaging_whole_brain(spikes_whole_brain,time_bins,stim):

    #then I have to build a dataframe of timebins versus neurons
    spikes_table = pd.DataFrame(0,columns=time_bins[:-1],index=['whole_brain'])

    # Now I should assign 1 to units which in a specific time bin have spiked  
    
    for t in range(len(time_bins)-1):
        br_spikes_number = get_num_of_spikes_in_bins_whole_brain(time_bins[t],time_bins[t+1],spikes_whole_brain,stim)
        spikes_table.loc[ 'whole_brain', time_bins[t]] = br_spikes_number

    return spikes_table


def encode_generated_dataframe_whole_brain(spikes_table):

    spatial_temporal_encoded_all_bins = np.zeros((1,D))
    #we have to slice during of each stimulation to 0.01 s and see which units spike in this interval
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    encoded_time_hypervector = generate_time_hypervector(bin_,D)
    for t in range(len(time_bins)-1):
        spatial_temporal_encoded_all_bins += encoded_time_hypervector[t] * int(spikes_table[time_bins[t]])

    return spatial_temporal_encoded_all_bins

def encode_generated_dataframe(spikes_table,encoded_all_units):
    
    spatial_temporal_encoded_all_bins = np.zeros((1,D))
    #we have to slice during of each stimulation to 0.01 s and see which units spike in this interval
    time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
    encoded_time_hypervector = generate_time_hypervector(bin_,D)
    for t in range(len(time_bins)-1):
        
        Unit_IDs_in_bin = []
        for index, value in spikes_table[time_bins[t]].iteritems():
            if value != 0:
                Unit_IDs_in_bin.extend([index] * value)
        Unit_IDs_in_bin = np.array(Unit_IDs_in_bin)

        sum_encoded_units = make_HV_of_units_in_binـillumination_method(encoded_all_units,Unit_IDs_in_bin)
        spatial_temporal_encoded_all_bins += encoded_time_hypervector[t] * sum_encoded_units
    return spatial_temporal_encoded_all_bins

################################################################################################################### fano factor : diff bins

################################################################################################################### 


##################################################################################  natural scenes 


def processing_and_augmenting_train_data(args): 
    
    frame,spatial_averaging_level = args

    Test_encoded_data = []
    Train_encoded_data = []
    ML_inputs_train = []
    ML_outputs_train = []
    ML_inputs_test = []
    ML_outputs_test = []
    Test_To_Shuffle = []
    
    list_of_stim_same_frame = list(grouped_by_frame.get_group(frame))
    train_data_each_frame_to_shuffle = [] #np.zeros((len(list_of_stim_same_pos),1))
    test_data_each_frame_to_shuffle = []

    random_index_test = random.sample(list_of_stim_same_frame, sample_size)
    random_index_train = [x for x in list_of_stim_same_frame if x not in random_index_test]

    
    for stim in list_of_stim_same_frame:
#         print('-----stim-----',stim)
#         tic = time.time()
        # Now I should assign 1 to units which in a specific time bin have spiked 
        if spatial_averaging_level == 'neuron_level':
            info_unit_spike = spikes[spikes['stimulus_presentation_id']==stim]
            spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
            
        
        elif spatial_averaging_level == 'population_level':
            spikes_table = spikes_bins_table_spatial_averaging_EX_IN(brain_regions,br_EX_IN_units,time_bins,stim)
            
            
        elif spatial_averaging_level == 'each_region_level':
            spikes_table = spikes_bins_table_spatial_averaging(brain_regions,time_bins,stim)
            
            
        elif spatial_averaging_level == 'super_region_level':
            spikes_table = spikes_bins_table_spatial_averaging(brain_regions,time_bins,stim)
            
            
        elif spatial_averaging_level == 'whole_brain':
            spikes_table = spikes_bins_table_spatial_averaging_whole_brain(spikes_whole_brain,time_bins,stim)
            

        vectorized_spikes_table = np.hstack(spikes_table.to_numpy())

        if stim in random_index_test:
            ML_inputs_test.append(vectorized_spikes_table)
            ML_outputs_test.append(frame)
            test_data_each_frame_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe(spikes_table,encoded_all_units)
            Test_encoded_data.append([frame,encoded_data_for_each_stim])
        else :
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(frame)

            train_data_each_frame_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe(spikes_table,encoded_all_units)

            Train_encoded_data.append([frame,encoded_data_for_each_stim])
    train_data_each_pos_to_shuffle = np.array(train_data_each_frame_to_shuffle)
    test_data_each_pos_to_shuffle = np.array(test_data_each_frame_to_shuffle)
    Test_To_Shuffle.append(test_data_each_pos_to_shuffle )
#     print('Test_To_Shuffle',np.array(Test_To_Shuffle).shape)
    # Shuffling the Train data
    for shuffle_time in range(shuffle_times_train):

        shuffled_train_data_each_pos = shuffle_nueros_spikes_in_stims(train_data_each_pos_to_shuffle)

        list_of_generated_stim_train = list(np.arange(shuffled_train_data_each_pos.shape[0]))

        for i in list_of_generated_stim_train:
            reshaped_array = shuffled_train_data_each_pos[i].reshape(spikes_table.shape)
            generated_spikes_table = pd.DataFrame(reshaped_array,index=spikes_table.index, columns=spikes_table.columns)
            # Now I should do something with this dataframe (encoding it) (like spatioal_temporal_encoding function)
            encoded_data_for_each_stim = encode_generated_dataframe(generated_spikes_table,encoded_all_units)
            vectorized_spikes_table = np.hstack(generated_spikes_table.to_numpy())
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(frame)

            Train_encoded_data.append([frame,encoded_data_for_each_stim])
    return     Train_encoded_data ,Test_encoded_data  ,ML_inputs_train ,ML_outputs_train ,ML_inputs_test ,ML_outputs_test ,Test_To_Shuffle 


def New_data_augmentation_shuffling_method(brain_regions,shuffle_times_train,spatial_averaging_level):

    tic = time.time() 
    spatial_averaging_level_vector = np.repeat(spatial_averaging_level, 118)
    print("start preparing data")
    #_________________________________
    
    with Pool(32) as p:
     
        Test_encoded_data = []
        Train_encoded_data = []
        ML_inputs_train = []
        ML_outputs_train = []
        ML_inputs_test = []
        ML_outputs_test = []
        Test_To_Shuffle = []

        results = p.imap(processing_and_augmenting_train_data, zip(List_of_frames,spatial_averaging_level_vector))
        for train_encoded_data, test_encoded_data   ,ml_inputs_train ,ml_outputs_train ,ml_inputs_test ,ml_outputs_test ,test_to_shuffle  in results:
            Train_encoded_data.extend(train_encoded_data) 
            Test_encoded_data.extend(test_encoded_data)
            ML_inputs_train.extend(ml_inputs_train)
            ML_outputs_train.extend(ml_outputs_train)
            ML_inputs_test.extend(ml_inputs_test)
            ML_outputs_test.extend(ml_outputs_test)
            Test_To_Shuffle.append(test_to_shuffle)
            
    
    Test_To_Shuffle  = np.squeeze(Test_To_Shuffle,axis =1)
    return Train_encoded_data, Test_encoded_data , ML_inputs_train ,ML_outputs_train  , ML_inputs_test , ML_outputs_test, Test_To_Shuffle


def Training(epochs,shuffle_times_test,Test_encoded_data):
    random.shuffle(Train_encoded_data)
    random.shuffle(Test_encoded_data)


    print("Training HDC ")
    F1_HDC = []
    F1_RF = []
    Classes = np.zeros((len(List_of_frames),D))

    for epoch in tqdm(range(epochs)):
        
        
        # Train HDC Model
        for i in range(len(Train_encoded_data)):
            frame = Train_encoded_data[i][0]
            data = Train_encoded_data[i][1]


            deltas,predicted_frame = check_similarity_to_all_classes_and_get_max_efficient(Classes,data)
            Classes = update_classes_new(Classes,deltas,data,frame,predicted_frame)

        print("train")
        y_true_train , y_pred_train = Test_model_shuffling_new(Classes,Train_encoded_data)
        print("micro-averaging")        
        averaging_metrics(y_true_train ,y_pred_train,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_train ,y_pred_train,'macro')
        

    for shuffle in range(shuffle_times_test):

        y_true_test_HDC , y_pred_test_HDC = Test_model_shuffling_new(Classes,Test_encoded_data)#zizizizi
        precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(y_true_test_HDC , y_pred_test_HDC )

        print("test")

        print("micro-averaging")
        averaging_metrics(y_true_test_HDC ,y_pred_test_HDC,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_test_HDC ,y_pred_test_HDC,'macro')
        
        F1_HDC.extend(f1_HDC)

        ML_inputs_test = []
        ML_outputs_test = []
        Test_encoded_data = []

        for frame in List_of_frames:
            test_data_each_pos_to_shuffle = np.array(Test_To_Shuffle[frame] )

            shuffled_test_data_each_pos = shuffle_nueros_spikes_in_stims(test_data_each_pos_to_shuffle)
            list_of_generated_stim_test = list(np.arange(shuffled_test_data_each_pos.shape[0]))

            for i in list_of_generated_stim_test:

                reshaped_array = shuffled_test_data_each_pos[i]
                generated_spikes_table = pd.DataFrame(reshaped_array,index=list(encoded_all_units.keys()), columns=time_bins[:-1])
                encoded_data_for_each_stim = encode_generated_dataframe(generated_spikes_table,encoded_all_units)
                vectorized_spikes_table = np.hstack(reshaped_array)
                Test_encoded_data.append([frame,encoded_data_for_each_stim])
                Test_encoded_data_all.append([frame,encoded_data_for_each_stim])
                ML_inputs_test_all.append(vectorized_spikes_table)
                ML_inputs_test.append(vectorized_spikes_table)
                ML_outputs_test.append(frame)
                ML_outputs_test_all.append(frame)

    return F1_HDC , Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all 


def Adaptive_training_data_augmentation_shuffling_efficient(Train_data,Test_data,brain_region,List_of_positions,epochs,suffle_times):
    Train_sample_each_pos=(suffle_times+1)*(45 - sample_size)
    Test_sample_each_pos = (suffle_times+1)*sample_size
    
    random.shuffle(Train_data)
    random.shuffle(Test_data)
    
    Classes = np.zeros((len(List_of_positions),D))
    # training model
    for epoch in tqdm(range(epochs), ascii=True,desc='training on epochs'):
        tic = time.time()
        for i in range(len(Train_data)):

            frame = Train_data[i][0]
            data = Train_data[i][1]

            deltas,predicted_frame = check_similarity_to_all_classes_and_get_max_efficient(Classes,data)
            Classes = update_classes_new(Classes,deltas,data,frame,predicted_frame)


        print("________________epoch________________ :  ",epoch)
        print("train")
        y_true_train , y_pred_train = Test_model_shuffling_new(Classes,Train_data)
        print("micro-averaging")        
        averaging_metrics(y_true_train ,y_pred_train,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_train ,y_pred_train,'macro')
        

        print("test")
        y_true_test , y_pred_test = Test_model_shuffling_new(Classes,Test_data)
        print("micro-averaging")
        averaging_metrics(y_true_test ,y_pred_test,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_test ,y_pred_test,'macro')
        
        precision, recall, f1_test, _ = precision_recall_fscore_support(y_true_test ,y_pred_test)
        tac = time.time()
        print("time : ",tac-tic)
    plot_heatmap(y_true_train , y_pred_train )
    plot_heatmap(y_true_test , y_pred_test )
    return f1_test

def check_similarity_to_all_classes_and_get_max_efficient(Classes,data):#salam

    results = np.matmul(Classes,data.reshape((D,1)))
    class_data_norm = LA.norm(Classes, axis=1) * LA.norm(data)     
    class_data_norm = class_data_norm.reshape((len(Classes), 1))
    delta = np.where(class_data_norm == 0, 0, results / class_data_norm)
    max_value = np.max(delta)
    indices = np.where(delta == max_value)[0]
    random_argmax = random.choice(indices)
    return delta,random_argmax

def update_classes_new(Classes,deltas,encoded_data_stim,frame,predicted_frame):
    if frame == predicted_frame:

        temp = eta_center*(1 - deltas[frame])*encoded_data_stim
        Classes[frame] += temp.reshape(D)
                
    else:
        temp1= eta_center*(deltas[predicted_frame] - deltas[frame])*encoded_data_stim
        Classes[predicted_frame] -= temp1.reshape(D)
        temp2 = eta_center*encoded_data_stim
        Classes[frame] += temp2.reshape(D)
    return Classes

def Test_model_shuffling_new(Classes,data):
    y_true = np.stack([row[0] for row in data])
    print('Classes',Classes.shape)
    rearrenged_data = np.squeeze(np.transpose(np.stack([row[1] for row in data])),axis=1)
    print('rearrenged_data',rearrenged_data.shape)
    class_samples = np.matmul(Classes,rearrenged_data)
    class_norm = LA.norm(Classes,axis = 1).reshape(len(Classes),1)
    data_norm = LA.norm(rearrenged_data,axis = 0).reshape(1,rearrenged_data.shape[1])
    class_data_norm = np.matmul(class_norm,data_norm)
    similarity_all_data = class_samples/class_data_norm
    y_pred = np.argmax(similarity_all_data, axis=0)
    
    return y_true,y_pred


def Test_model_shuffling(Classes,Train_data):
    y_true = []
    y_pred = [ ]
    
    for i in range(len(Train_data)):
        frame = Train_data[i][0]
        data = Train_data[i][1]
        deltas = check_similarity_to_all_classes(Classes,data)
        predicted_class=max(deltas, key = deltas.get)
        y_true.append(frame)
        y_pred.append(predicted_class)
    return y_true,y_pred


def update_classes_frame(Classes,deltas,encoded_data_stim,frame,predicted_frame):
    if frame == predicted_frame:
        Classes[frame] += eta_center*(1 - deltas[frame])*encoded_data_stim
        
    else:
        Classes[predicted_frame] -= eta_center*(deltas[predicted_frame] - deltas[frame])*encoded_data_stim
        Classes[frame] += eta_center*encoded_data_stim
        
    return Classes


def Training_(epochs,shuffle_times_test,Test_encoded_data_all ):  
    F1_HDC = []
    Classes = make_classes(List_of_frames)
    for epoch in tqdm(range(epochs)):
        
        print("________________epoch________________ :  ",epoch)

        # Train HDC Model
        for i in range(len(Train_encoded_data)):
            frame = Train_encoded_data[i][0]
            data = Train_encoded_data[i][1].astype('float64')
            deltas = check_similarity_to_all_classes(Classes,data)
            predicted_frame = give_max_frame_randomly(deltas)
            Classes = update_classes_frame(Classes,deltas,data,frame,predicted_frame)
            

        y_true_test_HDC , y_pred_test_HDC = Test_model_shuffling(Classes,Train_encoded_data)
        precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(y_true_test_HDC , y_pred_test_HDC )
        precision_HDC_micro, recall_HDC_micro, f1_HDC_micro = averaging_metrics(y_true_test_HDC , y_pred_test_HDC ,average='micro')
        precision_HDC_macro, recall_HDC_macro, f1_HDC_macro = averaging_metrics(y_true_test_HDC , y_pred_test_HDC ,average='macro')
        F1_HDC.append(f1_HDC)


    Test_encoded_data = Test_encoded_data_all 
    Test_encoded_data_all = list( Test_encoded_data_all )
    F1_HDC = []
    F1_RF = []
    for shuffle in range(shuffle_times_test):

        y_true_test_HDC , y_pred_test_HDC = Test_model_shuffling(Classes,Test_encoded_data)#zizizizi
        precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(y_true_test_HDC , y_pred_test_HDC )

        print("test")

        print("micro-averaging")
        averaging_metrics(y_true_test_HDC ,y_pred_test_HDC,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_test_HDC ,y_pred_test_HDC,'macro')
        
        F1_HDC.extend(f1_HDC)


        ML_inputs_test = []
        ML_outputs_test = []
        Test_encoded_data = []


        for frame in List_of_frames:
            test_data_each_pos_to_shuffle = np.array(Test_To_Shuffle[frame] )
            shuffled_test_data_each_pos = shuffle_nueros_spikes_in_stims(test_data_each_pos_to_shuffle)
            list_of_generated_stim_test = list(np.arange(shuffled_test_data_each_pos.shape[0]))

            for i in list_of_generated_stim_test:

                reshaped_array = shuffled_test_data_each_pos[i]

                encoded_data_for_each_stim = encode_generated_dataframe_new(reshaped_array,encoded_all_units)
                vectorized_spikes_table = np.hstack(reshaped_array)
                Test_encoded_data.append([frame,encoded_data_for_each_stim])
                ML_inputs_test_all.append(vectorized_spikes_table)
                ML_inputs_test.append(vectorized_spikes_table)
                ML_outputs_test.append(frame)
                ML_outputs_test_all.append(frame)
                
    return F1_HDC , Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all 


# # br_EX_IN_units = np.load(os.path.join(br_EX_IN_units_path, 'br_EX_IN_units.npy'),allow_pickle=True)
# # br_EX_IN_units = br_EX_IN_units.item()


# # whole_brain


print(' fano factor : diff bins')

for session_id in [799864342,732592105,754312389,754829445,757216464,757970808,759883607,763673393]:

    print('session_id',session_id)
    session_directory = os.path.join('wild_mice/', str(session_id)+'_test')
    if not os.path.exists(session_directory):
        os.mkdir(session_directory)
    

    session = cache.get_session_data(session_id)
    rf_mapping = ReceptiveFieldMapping(session)  
    
    
    epochs = 2
    shuffle_times = 5
    shuffle_times_train = 4
    shuffle_times_test = 4
    encoder = Encoder(features,D)

        
    session_natural_scenes_directory = os.path.join(session_directory, 'natural_scenes')
    if not os.path.exists(session_natural_scenes_directory):
        os.mkdir(session_natural_scenes_directory)
    
    rf_stim_table = session.stimulus_presentations[session.stimulus_presentations.stimulus_name == 'natural_scenes']  
    rf_stim_table['stimulus_presentation_id'] = rf_stim_table.index.values
    
    scene_presentations = session.get_stimulus_table("natural_scenes")  


    natural_scenes_spatial_averaging_directory = os.path.join(session_natural_scenes_directory, 'spatial_averaging')
    if not os.path.exists(natural_scenes_spatial_averaging_directory):
        os.mkdir(natural_scenes_spatial_averaging_directory)
            
    br_EX_IN_units_path = os.path.join(natural_scenes_spatial_averaging_directory,'population_level')
    if not os.path.exists(br_EX_IN_units_path):
        os.mkdir(br_EX_IN_units_path)
    
    List_of_frames = list(np.sort(rf_stim_table.frame.unique()))
    List_of_frames.pop(0)
    List_of_frames = list(np.array(List_of_frames).astype(int)) 
    List_of_positions = List_of_frames
    encoder = Encoder(features,D)
    grouped_by_frame = rf_stim_table.groupby('frame')["stimulus_presentation_id"]
    averaging_level = 'population_level'
    brain_regions = whole_brain_regions_check(session.structure_acronyms)
    spikes = spikes_diff_brain_region(brain_regions)
    br_spikes = spikes_seperate_brain_region(brain_regions)
    encoded_all_units = encode_brain_regions_EX_IN_dict(brain_regions)

    res_fano={}
    for bin_ in [0.250,0.125,0.050,0.025,0.010,0.005,0.002]:
        print('---------------------bin---------------------',bin_,'session_id',session_id )
        time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
        br_EX_IN_units = brain_region_EX_IN_unitsـFano_Factor_Natural_scenes(brain_regions)
        
        br_EX_IN_units_path_bin = os.path.join(br_EX_IN_units_path,str(bin_))
        if not os.path.exists(br_EX_IN_units_path_bin):
            os.mkdir(br_EX_IN_units_path_bin)
        np.save(os.path.join(br_EX_IN_units_path_bin, 'br_EX_IN_units'), np.array(br_EX_IN_units))
        bin_ = 0.001#inja bia
        time_bins = np.linspace(0,0.25, int(0.25/bin_)+1)
        Train_encoded_data, Test_encoded_data_all , ML_inputs_train ,ML_outputs_train  ,ML_inputs_test_all ,\
        ML_outputs_test_all ,Test_To_Shuffle=  New_data_augmentation_shuffling_method(brain_regions,shuffle_times_train,averaging_level)
        Test_encoded_data = Test_encoded_data_all
        F1_HDC_population_level , Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all = Training(epochs,shuffle_times_test,Test_encoded_data )
        
        np.save(os.path.join(br_EX_IN_units_path_bin, 'result.npy'), np.array(F1_HDC_population_level))

    
    # # br_EX_IN_units = np.load(os.path.join(br_EX_IN_units_path, 'br_EX_IN_units.npy'),allow_pickle=True)
    # # br_EX_IN_units = br_EX_IN_units.item()
    
    
