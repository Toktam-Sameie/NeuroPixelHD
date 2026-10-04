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
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from common import (
    Encoder,
    Multiprocessing,
    Threshhoding_receptive_field,
    averaging_metrics,
    check_in_region,
    check_similarity_to_all_classes,
    euclidean_distance,
    generate_random_vecotor,
    generate_time_slice_for_overlapping_bin,
    get_num_of_spikes_in_bins_whole_brain,
    get_unitID_in_bin,
    give_max_frame_randomly,
    give_max_position_randomly,
    interpolate_time_hypervectors_window,
    make_HV_of_units_in_binـillumination_method,
    make_accuracy_dict,
    make_classes,
    plot_heatmap_of_region_accuracy,
    save_all_inputs_to_directory,
    save_all_inputs_to_directory_natural_scenes,
    save_inputs_to_directory,
    save_result_to_directory,
    shuffle_nueros_spikes_in_stims,
    spikes_bins_table,
    spikes_bins_table_overlapping,
    update_classes,
)

random.seed(42)
np.random.seed(42)

output_dir = 'ecephys_cache_dir/'
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions_wild = brain_observatory_type_sessions[brain_observatory_type_sessions["full_genotype"] == 'wt/wt']

list_of_sessions_wild_type = list(brain_observatory_type_sessions_wild.index)


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

# previously, we had brain regions just to POL
eligible_brain_regions = ['VISp','VISam', 'VISal', 'VISrl', 'VISmma','VISpm', 'VISl','CA1', 'CA2', 'CA3','SUB', 'ProS','DG','TH', 'LP', 'LGd', 'PP', 'PIL', 'MGv', 'PO', 'Eth', 'POL']
#,'APN','ZI'
# 'LGv', 'VPL', 'IGL',' SGN', 'VPN', 'LD', 'POT','MRN', 'MB', 'NOT','PPT','OP','SCig'


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
        Y_num.append(encode_pos[tuple(target_pos[i])])
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


def random_forest(nestimators,Train_input , Train_target , Test_input , Test_target ):
    

    print(" Training RF")
    tic = time.time()
    rfc = RandomForestClassifier(n_estimators=nestimators)

    rfc.fit(Train_input, list(Train_target))
    print('done fitting with RF')
    
    
    y_pred_test = rfc.predict(Test_input)
    
    precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(Test_target , y_pred_test )
    
    print("___Test___")
    print("micro-averaging")
    precision, recall, f1_micro_test= averaging_metrics(Test_target ,y_pred_test,'micro')
    print("macro-averaging")
    precision, recall, f1_macro_test = averaging_metrics(Test_target ,y_pred_test,'macro')
    
    tac = time.time()
    y_pred_train = rfc.predict(Train_input)
    print("___Train____")
    print("micro-averaging")
    averaging_metrics(Train_target ,y_pred_train,'micro')
    print("macro-averaging")
    averaging_metrics(Train_target ,y_pred_train,'macro')
    

    return  f1_HDC ,tac - tic,


def neural_network(hls,Train_input , Train_target , Test_input , Test_target):
    
    from sklearn.neural_network import MLPClassifier

    print("Training Neural Network")
    tic = time.time()

    # Create the neural network classifier
    nn_classifier = MLPClassifier(hidden_layer_sizes=(hls,), activation='relu', solver='adam', random_state=42)


    nn_classifier.fit(Train_input, Train_target)
    print('Done fitting with the neural network')

    
    y_pred_test = nn_classifier.predict(Test_input)
    
    precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(Test_target , y_pred_test )
    
    print("___Test___")
    print("micro-averaging")
    precision, recall, f1_micro_test = averaging_metrics(Test_target ,y_pred_test,'micro')
    print("macro-averaging")
    precision, recall, f1_macro_test = averaging_metrics(Test_target ,y_pred_test,'macro')

    
    tac = time.time()
    y_pred_train = nn_classifier.predict(Train_input)
    print("___Train____")
    print("micro-averaging")
    averaging_metrics(Train_target ,y_pred_train,'micro')
    print("macro-averaging")
    averaging_metrics(Train_target ,y_pred_train,'macro')
    

    return f1_HDC , tac - tic


# Assuming ML_inputs_train, ML_inputs_test, Train_target, ML_outputs_test, label_to_tuple_pos, eucli_dis are defined elsewhere


def knn(k, Train_input, Train_target, Test_input, Test_target):
 
    
    print("Training K-Nearest Neighbors Classifier")
    tic = time.time()

    # Create the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    knn_classifier.fit(Train_input, Train_target)
    print('Done fitting with K-Nearest Neighbors Classifier')

    
    y_pred_test = knn_classifier.predict(Test_input)
    precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(Test_target , y_pred_test )
    print("___Test___")
    print("micro-averaging")
    precision, recall, f1_micro_test = averaging_metrics(Test_target, y_pred_test, 'micro')
    print("macro-averaging")
    precision, recall, f1_macro_test = averaging_metrics(Test_target, y_pred_test, 'macro')

    tac = time.time()
    y_pred_train = knn_classifier.predict(Train_input)
    print("___Train____")
    print("micro-averaging")
    averaging_metrics(Train_target, y_pred_train, 'micro')
    print("macro-averaging")
    averaging_metrics(Train_target, y_pred_train, 'macro')

    return f1_HDC , tac - tic


def naive_bayes(Train_input, Train_target, Test_input, Test_target):

    
    print("Training Naive Bayes Classifier")
    tic = time.time()

    # Create the Naive Bayes classifier
    nb_classifier = GaussianNB()

    nb_classifier.fit(Train_input, Train_target)
    print('Done fitting with Naive Bayes Classifier')

    tac = time.time()

    
    y_pred_test = nb_classifier.predict(Test_input)
    precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(Test_target , y_pred_test )
    print("___Test___")
    print("micro-averaging")
    precision, recall, f1_micro_test = averaging_metrics(Test_target, y_pred_test, 'micro')
    print("macro-averaging")
    precision, recall, f1_macro_test = averaging_metrics(Test_target, y_pred_test, 'macro')

    tac = time.time()
    y_pred_train = nb_classifier.predict(Train_input)
    print("___Train____")
    print("micro-averaging")
    averaging_metrics(Train_target, y_pred_train, 'micro')
    print("macro-averaging")
    averaging_metrics(Train_target, y_pred_train, 'macro')

    return f1_HDC , tac - tic


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
    if crop_overlap_mood == 'crop': 
        num_of_HV = int(np.floor(0.25/bin_))
    else:
        num_of_HV = int(np.ceil(0.25/bin_))
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
        if crop_overlap_mood == 'crop':
            spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
        elif crop_overlap_mood == 'overlap':
            spikes_table = spikes_bins_table_overlapping(info_unit_spike,time_bins,unit_IDs)
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
    return spatial_temporal_encoded_all_bins 


def Adaptive_training_data_augmentation_shuffling(Train_data,Test_data,brain_region,List_of_positions,epochs,suffle_times):
    
    Train_sample_each_pos=(suffle_times+1)*(45 - sample_size)
    Test_sample_each_pos = (suffle_times+1)*sample_size
    
    random.shuffle(Train_data)
    random.shuffle(Test_data)
    Classes = make_classes(List_of_positions)
    # training model
    tic = time.time()
    for epoch in tqdm(range(epochs), ascii=True,desc='training on epochs'):
        print("________________epoch________________ :  ",epoch)
        for i in range(len(Train_data)):

            pos = Train_data[i][0]
            data = Train_data[i][1].astype(np.float64)
            deltas = check_similarity_to_all_classes(Classes,data)
            predicted_class = give_max_position_randomly(deltas)
            in_region = check_in_region(pos,predicted_class)
            Classes = update_classes(Classes,deltas,data,pos,deltas[pos],predicted_class,deltas[predicted_class],in_region)

    
    tac = time.time()
    print('------------------final result------------------')
    accuarcy_test, euclidean_dis_test = Test_model_in_region_accuracy(Classes,Test_data,Test_sample_each_pos)
    return euclidean_dis_test , tac-tic

def new_update_gabor(Classes,data,pos,flat_pos,tuple_predicted_pos,flat_predicted_pos,delta_pos, delta_predicted_pos ,in_region):
    flat_neighbors_position_correct = make_neighbor_position_new(pos,flat_pos)
    if in_region: # if we predict the class correctly
        
        Classes[flat_neighbors_position_correct, :] += eta_region*(1 - delta_pos)*data
        Classes[flat_pos, :] += eta_center*(1 - delta_pos)*data
    else: # if we predict the class incorrectly
        # Update correct class
        Classes[flat_neighbors_position_correct, :] += eta_region*(1 - delta_pos)*data
        Classes[flat_pos, :] += eta_center*(delta_predicted_pos - delta_pos)*data
        flat_neighbors_position_incorrect = make_neighbor_position_new(tuple_predicted_pos,flat_predicted_pos)
        
        Classes[flat_neighbors_position_incorrect, :] -= eta_region*(delta_predicted_pos - delta_pos)*data
        Classes[flat_predicted_pos, :] -= eta_center*(delta_predicted_pos - delta_pos)*data       
    return Classes
        
def make_neighbor_position_new(pos,flat_pos):
    flat_neighbors_position = []
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if pos[0]+i >= 0 and pos[1]+j >= 0 and pos[0]+i < 9 and pos[1]+j < 9: 
                flat_neighbors_position.append(encode_pos[(pos[0]+i,pos[1]+j)])
    flat_neighbors_position.remove(flat_pos)
    return  flat_neighbors_position


def Test_model_efficient_gabor_new(Classes,data):
    tic = time.time()
    y_true_tuple = np.stack([row[0] for row in data])
    y_true_flat = np.stack([encode_pos[row[0]] for row in data])
    print('Classes',Classes.shape)
    rearrenged_data = np.transpose(np.stack([row[1] for row in data]))
    print('rearrenged_data',rearrenged_data.shape)
    class_samples = np.matmul(Classes,rearrenged_data)
    class_norm = LA.norm(Classes,axis = 1).reshape(len(Classes),1)
    data_norm = LA.norm(rearrenged_data,axis = 0).reshape(1,rearrenged_data.shape[1])
    class_data_norm = np.matmul(class_norm,data_norm)
    similarity_all_data = class_samples/class_data_norm
    y_pred = np.argmax(similarity_all_data, axis=0)
    y_pred_tuple = [decode_pos[i] for i in y_pred]
    euclidean_dis = np.linalg.norm(np.array(y_pred_tuple) - np.array(y_true_tuple), axis=1)
    tac = time.time()
    return euclidean_dis , tac-tic

def Adaptive_training_gabor_efficient(Train_data,Test_data,brain_region,List_of_positions,epochs,suffle_times):
    
    random.shuffle(Train_data)
    random.shuffle(Test_data)
    
    Classes = np.zeros((81,D)) # turn it to matrix
    # training model
    tic = time.time()
    for epoch in tqdm(range(2), ascii=True,desc='training on epochs'):
        
        for i in range(len(Train_data)):

            pos = Train_data[i][0]
            data = Train_data[i][1]
            pos = tuple(int(element) for element in pos)
            flat_pos = encode_pos[pos]
            similarity_scores = cosine_similarity(data.reshape(1, -1), Classes)
            similarity_scores = similarity_scores.flatten()
            max_value = np.max(similarity_scores)
            indices = np.where(similarity_scores == max_value)[0]
            flat_predicted_pos = random.choice(indices)
            # Now I have to unravel random_argmax
#             tuple_predicted_pos = np.unravel_index(flat_predicted_pos,(9,9))
            tuple_predicted_pos = decode_pos[flat_predicted_pos]
            delta_pos , delta_predicted_pos = similarity_scores[flat_pos] , max_value
            # check in region 
            in_region = check_in_region(pos,tuple_predicted_pos)
            # update classes
            Classes = new_update_gabor(Classes,data,pos,flat_pos,tuple_predicted_pos,flat_predicted_pos,delta_pos, delta_predicted_pos ,in_region)


    euclidean_dis_test , t_test = Test_model_efficient_gabor_new(Classes,Test_data)
    tac = time.time()
    return euclidean_dis_test , tac-tic


def augmenting_data_plus_training_temporal_averaging(bin_,brain_regions,epochs,shuffle_times):

    Train_encoded_data , Test_encoded_data , ML_inputs_train , ML_outputs_train , ML_inputs_test ,\
    ML_outputs_test = Multiprocessing(data_augmentation_shuffling_method_neuron_level,List_of_positions)
    
    euclidean_dis , accuarcy_test  =  Adaptive_training_data_augmentation_shuffling(Train_encoded_data,Test_encoded_data,brain_regions,List_of_positions,epochs,shuffle_times)
    
    save_inputs_to_directory(euclidean_dis,gabor_temporal_averaging_directory,str(bin_))
    return accuarcy_test 


def get_data_bin(session_id,bin_,Train_encoded_data_or, Test_encoded_data_or , ML_inputs_train_or , ML_outputs_train_or, ML_inputs_test_or, ML_outputs_test_or):
    
    Train_encoded_data= np.load( Train_encoded_data_or.replace('0.001', bin_),allow_pickle=True)
    Test_encoded_data = np.load( Test_encoded_data_or.replace('0.001', bin_),allow_pickle=True)
    ML_inputs_train = np.load( ML_inputs_train_or.replace('0.001', bin_),allow_pickle=True)
    ML_outputs_train = np.load( ML_outputs_train_or.replace('0.001', bin_),allow_pickle=True)
    ML_inputs_test = np.load( ML_inputs_test_or.replace('0.001', bin_),allow_pickle=True)
    ML_outputs_test= np.load( ML_outputs_test_or.replace('0.001', bin_),allow_pickle=True)
    return Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test

def get_data_level(session_id,level,Train_encoded_data_or, Test_encoded_data_or , ML_inputs_train_or , ML_outputs_train_or, ML_inputs_test_or, ML_outputs_test_or):
    
    Train_encoded_data= np.load( Train_encoded_data_or.replace('neuron_level', level),allow_pickle=True)
    Test_encoded_data = np.load( Test_encoded_data_or.replace('neuron_level', level),allow_pickle=True)
    ML_inputs_train = np.load( ML_inputs_train_or.replace('neuron_level', level),allow_pickle=True)
    ML_outputs_train = np.load( ML_outputs_train_or.replace('neuron_level', level),allow_pickle=True)
    ML_inputs_test = np.load( ML_inputs_test_or.replace('neuron_level', level),allow_pickle=True)
    ML_outputs_test= np.load( ML_outputs_test_or.replace('neuron_level', level),allow_pickle=True)
    return Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test

    
#         # Original_Train_encoded_data = get_not_augmented_samples_gabor(Train_encoded_data,35)
#         # ML_inputs_train = get_not_augmented_samples_gabor(ML_inputs_train,35)
#         # ML_outputs_train = get_not_augmented_samples_gabor(ML_outputs_train,35)
        
#         # Original_Test_encoded_data = get_not_augmented_samples_gabor(Test_encoded_data,10)
#         # ML_inputs_test = get_not_augmented_samples_gabor(ML_inputs_test,10)
#         # ML_outputs_test = get_not_augmented_samples_gabor(ML_outputs_test,10)
        
        
#         # euc_dis_test_HDC , time_HDC =  Adaptive_training_data_augmentation_shuffling(Original_Train_encoded_data,Original_Test_encoded_data,brain_regions,List_of_positions,epochs,shuffle_times)

#         # save_result_to_directory(euc_dis_test_HDC,'euc_dis_HDC',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_HDC,'time_HDC',gabor_temporal_averaging_directory,str(bin_))
        
#         # euc_dis_test_RF , time_RF = random_forest(10)
#         # save_result_to_directory(euc_dis_test_RF,'euc_dis_RF',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_RF,'time_RF',gabor_temporal_averaging_directory,str(bin_))
        
#         # euc_dis_test_svm , time_svm = svm('rbf',0.5)
#         # save_result_to_directory(euc_dis_test_svm,'euc_dis_svm',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_svm,'time_svm',gabor_temporal_averaging_directory,str(bin_))
    
        
#         #print("LDA")
#         #euc_dis_test_lda, time_lda = lda_algorithm()
#         #save_result_to_directory(euc_dis_test_lda,'euc_dis_lda',gabor_temporal_averaging_directory,str(bin_))
#         #save_result_to_directory(time_lda,'time_lda',gabor_temporal_averaging_directory,str(bin_))
    
#         # print("Logistic Regression")
#         # euc_dis_test_lr, time_lr = logistic_regression_algorithm()
#         # save_result_to_directory(euc_dis_test_lr,'euc_dis_lr',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_lr,'time_lr',gabor_temporal_averaging_directory,str(bin_))
    
#         # print("Naive Bayse")
#         # euc_dis_test_nb, time_nb = naive_bayes()
#         # save_result_to_directory(euc_dis_test_nb,'euc_dis_nb',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nb,'time_nb',gabor_temporal_averaging_directory,str(bin_))
    
#         # print("KNN")
#         # euc_dis_test_knn, time_knn = knn(k=18)
#         # save_result_to_directory(euc_dis_test_knn,'euc_dis_knn',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_knn,'time_knn',gabor_temporal_averaging_directory,str(bin_))
    
#         # print("gradient_boosting")
#         # euc_dis_test_gb, time_gb = gradient_boosting()
#         # save_result_to_directory(euc_dis_test_gb,'euc_dis_gb',gabor_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_gb,'time_gb',gabor_temporal_averaging_directory,str(bin_))
        
# what
################################################################################## Temporal Averaging new bins
# print('Temporal Averaging new bins')
# for session_id in [759883607,761418226]:

    
# for session_id in [757970808,759883607,761418226]:

    
 ##################################################################################  spatial averaging


def brain_region_EX_IN_unitsـFano_Factor_Gabor(brain_regions,bin_FF):
    print('brain_region_EX_IN_unitsـFano_Factor_Gabor')
    time_bins = np.linspace(0,0.25, int(0.25/bin_FF)+1)
    
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
    return br_EX_IN_units 


def spikes_bins_table_spatial_averaging_EX_IN(brain_regions,br_EX_IN_units,time_bins,stim):
    if crop_overlap_mood == 'crop':
        
        spikes_table = pd.DataFrame(columns=time_bins[:-1])
    
        for region in brain_regions:
            ex_row = pd.DataFrame(index=[f'{region}_EX'], columns=time_bins[:-1])
            in_row = pd.DataFrame(index=[f'{region}_IN'], columns=time_bins[:-1])

            EX_IN_units = br_EX_IN_units[region]
            print('EX_IN_units',EX_IN_units.keys())
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
                    if key in list(EX_IN_units.keys()):
                        
                        
                        if EX_IN_units[key] == 'excitatory':
                            sum_ex += value
        
                        elif EX_IN_units[key] == 'inhibitory':
                            sum_in += value
                    else:
                        sum_ex += value

                ex_row.loc[f'{region}_EX'][t] = sum_ex  
                in_row.loc[f'{region}_IN'][t] = sum_in  
            
            spikes_table = spikes_table.append(ex_row)
            spikes_table = spikes_table.append(in_row) 
            
    if crop_overlap_mood == 'overlap':
        
        spikes_table = pd.DataFrame(columns=time_bins[:,0])
        for region in brain_regions:

            ex_row = pd.DataFrame(index=[f'{region}_EX'], columns=time_bins[:,0])
            in_row = pd.DataFrame(index=[f'{region}_IN'], columns=time_bins[:,0])

            EX_IN_units = br_EX_IN_units[region]
            spikes_br = br_spikes[region]
            info_unit_spike = spikes_br[spikes_br['stimulus_presentation_id'] == stim]
            for n in range(len(time_bins)):
                condition = ((info_unit_spike['time_since_stimulus_presentation_onset'] < time_bins[n][1]) &( time_bins[n][0]< info_unit_spike['time_since_stimulus_presentation_onset']))
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

                ex_row.loc[f'{region}_EX'][n] = sum_ex  
                in_row.loc[f'{region}_IN'][n] = sum_in  
            
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

    if crop_overlap_mood == 'crop':
        spikes_table = pd.DataFrame(0,columns=time_bins[:-1], index=brain_regions)

        for t in range(len(time_bins)-1):
            br_spikes_number = get_num_of_spikes_in_bins(time_bins[t],time_bins[t+1],brain_regions,stim)
            for br, count in br_spikes_number.items(): 
                spikes_table.loc[br, time_bins[t]] = count

    
    elif crop_overlap_mood == 'overlap':
        spikes_table = pd.DataFrame(0,columns=time_bins[:,0], index=brain_regions)

        for n in range(len(time_bins)):
            br_spikes_number = get_num_of_spikes_in_bins(time_bins[n][0],time_bins[n][1],brain_regions,stim)
            for br, count in br_spikes_number.items(): 
                spikes_table.loc[br, time_bins[n][0]] = count
        
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
            encoded_all_units.append(brain_region_HV * encode_receptive_field(unitID,encoder,threshhold).reshape((1,D)))
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
            if crop_overlap_mood == 'crop':
                spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
            elif crop_overlap_mood == 'overlap':
                spikes_table = spikes_bins_table_overlapping(info_unit_spike,time_bins,unit_IDs)
            
        
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
            encoded_data_for_each_stim = encode_generated_dataframe_new(spikes_table,encoded_all_units)

            Test_encoded_data.append([pos,np.array(encoded_data_for_each_stim)])
        else :
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(pos)

            train_data_each_pos_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe_new(spikes_table,encoded_all_units)

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

    if crop_overlap_mood == 'crop':
        spikes_table = pd.DataFrame(0,columns=time_bins[:-1],index=['whole_brain'])

        for t in range(len(time_bins)-1):
            br_spikes_number = get_num_of_spikes_in_bins_whole_brain(time_bins[t],time_bins[t+1],spikes_whole_brain,stim)
            spikes_table.loc[ 'whole_brain', time_bins[t]] = br_spikes_number

    elif crop_overlap_mood == 'overlap':
        
        spikes_table = pd.DataFrame(0,columns=time_bins[:,0],index=['whole_brain'])
        
        for n in range(len(time_bins)):
            br_spikes_number = get_num_of_spikes_in_bins_whole_brain(time_bins[n][0],time_bins[n][1],spikes_whole_brain,stim)
            spikes_table.loc[ 'whole_brain', time_bins[n][0]] = br_spikes_number
            
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

def run_ML_model_in_one_block(directory,spatial_averaging_level):
    
        euc_dis_test_nn , time_nn = neural_network(10)
        save_result_to_directory(euc_dis_test_nn,'euc_dis_NN',directory,spatial_averaging_level)
        save_result_to_directory(time_nn,'time_nn',directory,spatial_averaging_level)

    
        euc_dis_test_HDC , time_HDC =  Adaptive_training_data_augmentation_shuffling(Train_encoded_data,Test_encoded_data,brain_regions,List_of_positions,epochs,shuffle_times)
        print('Time:', time_HDC)
        print('Mean of Euclidean distance test HDC:', np.mean(euc_dis_test_HDC))
        save_result_to_directory(euc_dis_test_HDC,'euc_dis_HDC',directory,spatial_averaging_level)
        save_result_to_directory(time_HDC,'time_HDC',directory,spatial_averaging_level)
        
        euc_dis_test_RF , time_RF = random_forest(10)
        save_result_to_directory(euc_dis_test_RF,'euc_dis_RF',directory,spatial_averaging_level)
        save_result_to_directory(time_RF,'time_RF',directory,spatial_averaging_level)

    
        euc_dis_test_nb, time_nb = naive_bayes()
        save_result_to_directory(euc_dis_test_nb,'euc_dis_nb',directory,spatial_averaging_level)
        save_result_to_directory(time_nb,'time_nb',directory,spatial_averaging_level)
    

        euc_dis_test_knn, time_knn = knn(k=18)
        save_result_to_directory(euc_dis_test_knn,'euc_dis_knn',directory,spatial_averaging_level)
        save_result_to_directory(time_knn,'time_knn',directory,spatial_averaging_level)
    

#     for bin_ in ['0.001','0.002','0.005','0.01','0.025','0.05','0.075','0.075overlap','0.1','0.1overlap','0.125','0.15','0.15overlap','0.2','0.2overlap','0.25'][::-1]:
        
        
# ################################################################################################################### 
    

original_str_001 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.001/euc_dis_HDC.npy'
original_str_002 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.002/euc_dis_HDC.npy'
original_str_005 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.005/euc_dis_HDC.npy'
original_str_010 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.01/euc_dis_HDC.npy'
original_str_025 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.025/euc_dis_HDC.npy'
original_str_050 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.05/euc_dis_HDC.npy'
original_str_075 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.075/euc_dis_HDC.npy'
original_str_075overlap = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.075overlap/euc_dis_HDC.npy'
original_str_100 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.1/euc_dis_HDC.npy'
original_str_100overlap = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.1overlap/euc_dis_HDC.npy'
original_str_125 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.125/euc_dis_HDC.npy'
original_str_150 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.15/euc_dis_HDC.npy'
original_str_150overlap = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.15overlap/euc_dis_HDC.npy'
original_str_200 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.2/euc_dis_HDC.npy'
original_str_200overlap = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.2overlap/euc_dis_HDC.npy'
original_str_250 = '/media/ssd/toktam/757216464/gabor/ML_Models_temporal_averaging/0.25/euc_dis_HDC.npy'


def optimal_time_bin(mouse_ID,new_alg):
    
    Original_str_001 = original_str_001.replace('757216464', mouse_ID)
    Original_str_002 = original_str_002.replace('757216464', mouse_ID)
    Original_str_005 = original_str_005.replace('757216464', mouse_ID)
    Original_str_010 = original_str_010.replace('757216464', mouse_ID)
    Original_str_025 = original_str_025.replace('757216464', mouse_ID)
    Original_str_050 = original_str_050.replace('757216464', mouse_ID)
    Original_str_075 = original_str_075.replace('757216464', mouse_ID)
    Original_str_075overlap = original_str_075overlap.replace('757216464', mouse_ID)
    Original_str_100 = original_str_100.replace('757216464', mouse_ID)
    Original_str_100overlap = original_str_100overlap.replace('757216464', mouse_ID)
    Original_str_125 = original_str_125.replace('757216464', mouse_ID)
    Original_str_150 = original_str_150.replace('757216464', mouse_ID)
    Original_str_150overlap = original_str_150overlap.replace('757216464', mouse_ID)
    Original_str_200 = original_str_200.replace('757216464', mouse_ID)
    Original_str_200overlap = original_str_200overlap.replace('757216464', mouse_ID)
    Original_str_250 = original_str_250.replace('757216464', mouse_ID)
    
    updated_str_001 = Original_str_001.replace('euc_dis_HDC', new_alg)
    updated_str_002 = Original_str_002.replace('euc_dis_HDC', new_alg)
    updated_str_005 = Original_str_005.replace('euc_dis_HDC', new_alg)
    updated_str_010 = Original_str_010.replace('euc_dis_HDC', new_alg)
    updated_str_025 = Original_str_025.replace('euc_dis_HDC', new_alg)
    updated_str_050 = Original_str_050.replace('euc_dis_HDC', new_alg)
    updated_str_075 = Original_str_075.replace('euc_dis_HDC', new_alg)
    updated_str_075overlap = Original_str_075overlap.replace('euc_dis_HDC', new_alg)
    updated_str_100 = Original_str_100.replace('euc_dis_HDC', new_alg)
    updated_str_100overlap = Original_str_100overlap.replace('euc_dis_HDC', new_alg)
    updated_str_125 = Original_str_125.replace('euc_dis_HDC', new_alg)
    updated_str_150 = Original_str_150.replace('euc_dis_HDC', new_alg)
    updated_str_150overlap = Original_str_150overlap.replace('euc_dis_HDC', new_alg)
    updated_str_200 = Original_str_200.replace('euc_dis_HDC', new_alg)
    updated_str_200overlap = Original_str_200overlap.replace('euc_dis_HDC', new_alg)
    updated_str_250 = Original_str_250.replace('euc_dis_HDC', new_alg)
    
    res_dict = {}

    res_dict[0.001] = np.load(updated_str_001,allow_pickle=True)
    res_dict[0.002] = np.load(updated_str_002,allow_pickle=True)
    res_dict[0.005] = np.load(updated_str_005,allow_pickle=True)
    res_dict[0.01] = np.load(updated_str_010,allow_pickle=True)
    res_dict[0.025] = np.load(updated_str_025,allow_pickle=True)
    res_dict[0.05]  = np.load(updated_str_050,allow_pickle=True)
    res_dict[0.075] = (np.load(updated_str_075,allow_pickle=True)+ np.load(updated_str_075overlap,allow_pickle=True))/2
    res_dict[0.1] = (np.load(updated_str_100,allow_pickle=True)+ np.load(updated_str_100overlap,allow_pickle=True))/2
    res_dict[0.125] = np.load(updated_str_125,allow_pickle=True)
    res_dict[0.15] = (np.load(updated_str_150,allow_pickle=True)+np.load(updated_str_150overlap,allow_pickle=True))/2
    res_dict[0.2] = (np.load(updated_str_200,allow_pickle=True)+np.load(updated_str_200overlap,allow_pickle=True))/2
    res_dict[0.25 ] = np.load(updated_str_250,allow_pickle=True)
    
    p_value_matrix = np.zeros((12,12))
    for i,bin_i in enumerate(list(res_dict.keys())):
        for j,bin_j in enumerate(list(res_dict.keys())):
            if i != j :
                p_value_matrix[i,j] = Wilcoxon_signed_rank(res_dict[bin_i],res_dict[bin_j],'greater')

    min_sum_row_index = np.argmin(np.sum(p_value_matrix, axis=1))
    bin_ = list(res_dict.keys())[min_sum_row_index ]
    return bin_

# #,,761418226
# for session_id in [754829445]:
    
#     # done : 757970808 , 757216464 ,763673393 , 759883607,799864342,732592105,754312389,
#     # crop (125) : 757216464 ,763673393  , 799864342 (just neuron and pop hdc and Rf),732592105(nothing)
#     #overlapped (200) : 757970808 , 759883607(should be runed again -> wrong crop overlapped label) 761418226(not completed) , [754312389 , 754829445]-> left
    
    
#     # bin_ = 0.2
    
 
#     #_______________________________________________________________________________________________________________________________________#

    
#     # Train_encoded_data_or = '/media/ssd/toktam/757970808/gabor/ML_Models_temporal_averaging/0.001/Train_encoded_data.npy'
#     # Test_encoded_data_or = '/media/ssd/toktam/757970808/gabor/ML_Models_temporal_averaging/0.001/Test_encoded_data.npy'
#     # ML_inputs_train_or = '/media/ssd/toktam/757970808/gabor/ML_Models_temporal_averaging/0.001/ML_inputs_train.npy'
#     # ML_outputs_train_or = '/media/ssd/toktam/757970808/gabor/ML_Models_temporal_averaging/0.001/ML_outputs_train.npy'
#     # ML_inputs_test_or = '/media/ssd/toktam/757970808/gabor/ML_Models_temporal_averaging/0.001/ML_inputs_test.npy'
#     # ML_outputs_test_or = '/media/ssd/toktam/757970808/gabor/ML_Models_temporal_averaging/0.001/ML_outputs_test.npy'
    
#     # Train_encoded_data_or  = Train_encoded_data_or.replace('757970808', str(session_id))
#     # Test_encoded_data_or = Test_encoded_data_or.replace('757970808', str(session_id))
#     # ML_inputs_train_or = ML_inputs_train_or.replace('757970808', str(session_id))
#     # ML_outputs_train_or = ML_outputs_train_or.replace('757970808', str(session_id))
#     # ML_inputs_test_or = ML_inputs_test_or.replace('757970808', str(session_id))
#     # ML_outputs_test_or = ML_outputs_test_or.replace('757970808', str(session_id))
    
#     # Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test= get_data_bin(session_id,str(bin_))

#     print('bin_',bin_)
#     Train_encoded_data, Test_encoded_data , ML_inputs_train ,ML_outputs_train  ,\
# ML_inputs_test , ML_outputs_test =  data_augmentation_shuffling_method(brain_regions,'neuron_level')
    
    
#     #_______________________________________________________________________________________________________________________________________#
    

#     # br_EX_IN_units = brain_region_EX_IN_unitsـFano_Factor_Gabor(brain_regions)
#     # br_EX_IN_units_path = os.path.join(gabor_spatial_averaging_directory,'population_level')
#     # if not os.path.exists(br_EX_IN_units_path):
#     #     os.mkdir(br_EX_IN_units_path)

#     # np.save(os.path.join(br_EX_IN_units_path, 'br_EX_IN_units'), np.array(br_EX_IN_units))


#     ############################### load generated data
#     # Train_encoded_data_or ='/media/ssd/toktam/757216464/gabor/ML_Models_spatial_averaging/population_level/Train_encoded_data.npy'
#     # Test_encoded_data_or = '/media/ssd/toktam/757216464/gabor/ML_Models_spatial_averaging/population_level/Test_encoded_data.npy'
#     # ML_inputs_train_or = '/media/ssd/toktam/757216464/gabor/ML_Models_spatial_averaging/population_level/ML_inputs_train.npy'
#     # ML_outputs_train_or = '/media/ssd/toktam/757216464/gabor/ML_Models_spatial_averaging/population_level/ML_outputs_train.npy'
#     # ML_inputs_test_or = '/media/ssd/toktam/757216464/gabor/ML_Models_spatial_averaging/population_level/ML_inputs_test.npy'
#     # ML_outputs_test_or = '/media/ssd/toktam/757216464/gabor/ML_Models_spatial_averaging/population_level/ML_outputs_test.npy'

#     # Train_encoded_data_or  = Train_encoded_data_or.replace('757216464', str(session_id))
#     # Test_encoded_data_or = Test_encoded_data_or.replace('757216464', str(session_id))
#     # ML_inputs_train_or = ML_inputs_train_or.replace('757216464', str(session_id))
#     # ML_outputs_train_or = ML_outputs_train_or.replace('757216464', str(session_id))
#     # ML_inputs_test_or = ML_inputs_test_or.replace('757216464', str(session_id))
#     # ML_outputs_test_or = ML_outputs_test_or.replace('757216464', str(session_id))

#     # Train_encoded_data= np.load( Train_encoded_data_or,allow_pickle=True)
#     # Test_encoded_data = np.load( Test_encoded_data_or,allow_pickle=True)
#     # ML_inputs_train = np.load( ML_inputs_train_or,allow_pickle=True)
#     # ML_outputs_train = np.load( ML_outputs_train_or,allow_pickle=True)
#     # ML_inputs_test = np.load( ML_inputs_test_or,allow_pickle=True)
#     # ML_outputs_test= np.load( ML_outputs_test_or,allow_pickle=True)
    
#     # Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test= get_data_bin(session_id,str(bin_))
#     ###############################
    
    
#     #_______________________________________________________________________________________________________________________________________#
    
    
#     #_______________________________________________________________________________________________________________________________________#


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
            if crop_overlap_mood == 'crop':
                spikes_table = spikes_bins_table(info_unit_spike,time_bins,unit_IDs)
            elif crop_overlap_mood == 'overlap':
                spikes_table = spikes_bins_table_overlapping(info_unit_spike,time_bins,unit_IDs)
                
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
            encoded_data_for_each_stim = encode_generated_dataframe_new(spikes_table,encoded_all_units)
            Test_encoded_data.append([frame,encoded_data_for_each_stim])
        else :
            ML_inputs_train.append(vectorized_spikes_table)
            ML_outputs_train.append(frame)

            train_data_each_frame_to_shuffle.append(spikes_table.values.tolist())
            encoded_data_for_each_stim = encode_generated_dataframe_new(spikes_table,encoded_all_units)

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
            encoded_data_for_each_stim = encode_generated_dataframe_new(generated_spikes_table,encoded_all_units)
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


def Training(epochs,shuffle_times_test,Train_encoded_data,Test_encoded_data, Test_encoded_data_all):
    random.shuffle(Train_encoded_data)
    random.shuffle(Test_encoded_data)


    print("Training HDC ")
    F1_HDC = []
    F1_RF = []
    Classes = np.zeros((len(List_of_frames),D)).astype('float64')
    time_all = []
    for epoch in tqdm(range(epochs)):
        tic = time.time()
        

        # Train HDC Model
        for i in range(len(Train_encoded_data)):
            frame = Train_encoded_data[i][0]
            data = np.array(Train_encoded_data[i][1]).astype('float64')


            deltas,predicted_frame = check_similarity_to_all_classes_and_get_max_efficient(Classes,data)
            Classes = update_classes_new(Classes,deltas,data,frame,predicted_frame)

        print("_________    train      _________")
        y_true_train , y_pred_train = Test_model_shuffling_new(Classes,Train_encoded_data)
        print("micro-averaging")        
        averaging_metrics(y_true_train ,y_pred_train,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_train ,y_pred_train,'macro')


    for shuffle in range(shuffle_times_test):

        y_true_test_HDC , y_pred_test_HDC = Test_model_shuffling_new(Classes,Test_encoded_data)#zizizizi
        precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(y_true_test_HDC , y_pred_test_HDC )
        
        tac = time.time()
        time_all.append(tac-tic)
        
        print("_________    test      _________")

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
                if crop_overlap_mood == 'crop':
                    generated_spikes_table = pd.DataFrame(reshaped_array,index=unit_IDs, columns=time_bins[:-1])
                elif crop_overlap_mood == 'overlap':
                    generated_spikes_table = pd.DataFrame(reshaped_array,index=unit_IDs, columns=time_bins[:,0])
                encoded_data_for_each_stim = encode_generated_dataframe_new(generated_spikes_table,encoded_all_units)
                vectorized_spikes_table = np.hstack(reshaped_array)
                Test_encoded_data.append([frame,encoded_data_for_each_stim])
                Test_encoded_data_all.append([frame,encoded_data_for_each_stim])
                ML_inputs_test_all.append(vectorized_spikes_table)
                ML_inputs_test.append(vectorized_spikes_table)
                ML_outputs_test.append(frame)
                ML_outputs_test_all.append(frame)

    return F1_HDC , Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all , sum(time_all)


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

    delta = np.divide(results, class_data_norm, out=np.zeros_like(results), where=class_data_norm!=0)


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
    y_true = np.stack([np.array(row[0] ).astype(float) for row in data])
    rearrenged_data = np.array(np.transpose(np.stack([row[1] for row in data]))).astype(float)
    print(np.isnan(rearrenged_data).any())
    print(np.isinf(rearrenged_data).any())
    class_samples = np.matmul(Classes,rearrenged_data)
    class_norm = LA.norm(Classes,axis = 1).reshape(len(Classes),1)
    data_norm = np.sqrt(np.sum(rearrenged_data**2, axis=0)).reshape(1,rearrenged_data.shape[1])
    class_data_norm = np.matmul(class_norm,data_norm)
    similarity_all_data = class_samples/class_data_norm
    y_pred = np.argmax(similarity_all_data, axis=0)
    
    return y_true,y_pred


def Test_model_shuffling(Classes,Train_data):
    y_true = []
    y_pred = [ ]
    
    for i in range(len(Train_data)):
        frame = Train_data[i][0]
        data = np.array(Train_data[i][1])
        deltas = check_similarity_to_all_classes(Classes,data)
        predicted_class=max(deltas, key = deltas.get)
        y_true.append(frame)
        y_pred.append(predicted_class)
    return y_true,y_pred


def update_classes_frame(Classes,deltas,encoded_data_stim,frame,predicted_frame):
    if frame == predicted_frame:

        temp = eta_center*(1 - deltas[frame])*encoded_data_stim
        Classes[frame] += temp.reshape(D)
                
    else:
        temp1= eta_center*(deltas[predicted_frame] - deltas[frame])*encoded_data_stim
        Classes[predicted_frame] -= temp1.reshape(D)
        temp2 = eta_center*encoded_data_stim
        Classes[frame] += temp2.reshape(D)
    return Classes

    
#     else:
# #         print("Classes[predicted_frame]",Classes[predicted_frame],Classes[predicted_frame].dtype)
#         print('deltas[predicted_frame]',deltas[predicted_frame])
#         print('deltas[frame]',deltas[frame])
#         print('eta_center',eta_center)
#         Classes[predicted_frame] -= eta_center*(deltas[predicted_frame] - deltas[frame])*encoded_data_stim
#         Classes[frame] += eta_center*encoded_data_stim
        

def Training_(epochs,shuffle_times_test,Train_encoded_data,Test_encoded_data,Test_encoded_data_all ):  
    F1_HDC = []
    Classes = make_classes(List_of_frames)
    time_all =[]
    for epoch in tqdm(range(epochs)):
        tic = time.time()
        print("________________epoch________________ :  ",epoch)

        # Train HDC Model
        for i in range(len(Train_encoded_data)):
            frame = Train_encoded_data[i][0]
            data = np.array(Train_encoded_data[i][1].astype('float64'))
            deltas = check_similarity_to_all_classes(Classes,data)
            predicted_frame = give_max_frame_randomly(deltas)
            Classes = update_classes_frame(Classes,deltas,data,frame,predicted_frame)
            

        tac = time.time()
        time_all.append(tac-tic)
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
        tic = time.time()
        
        y_true_test_HDC , y_pred_test_HDC = Test_model_shuffling(Classes,Test_encoded_data)#zizizizi
        precision_HDC, recall_HDC, f1_HDC, _ = precision_recall_fscore_support(y_true_test_HDC , y_pred_test_HDC )
        tac = time.time()
        time_all.append(tac-tic)
        print("test")

        print("micro-averaging")
        averaging_metrics(y_true_test_HDC ,y_pred_test_HDC,'micro')
        print("macro-averaging")
        averaging_metrics(y_true_test_HDC ,y_pred_test_HDC,'macro')
        
        F1_HDC.extend(f1_HDC)


        ML_inputs_test = []
        ML_outputs_test = []
        Test_encoded_data = []
        tac = time.time()
        time_all.append(tac-tic)


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
                
    return F1_HDC , Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all , sum(time_all)


################################################################################## temporal averaging
# print('Naural scenes -> temporal averaging') # incomplete
#757216464 , 759883607,761418226,799864342 done
# 757970808 just 0.175over left
# 763673393 got a problem
#754312389,754829445,732592105 not run


#757216464 ,757970808, 759883607,761418226,799864342 
# for session_id in [ 754312389]:

        
#     encoder = Encoder(features,D)
#     brain_regions = whole_brain_regions_check(session.structure_acronyms) #whole brain
#     encoded_all_units = encode_all_units_seperating_region(brain_regions)
#     unit_IDs = list(spikes.unit_id.unique())
#     shuffle_times_train = 4
#     shuffle_times_test = 4
#     epochs = 3
#     # done 0.175, 0.225
#     for bin_ in [  0.250, 0.200, 0.150,0.125, 0.100,0.075,0.050,0.025,0.010,0.005,0.002,0.001 ]:
        
#         # Train_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/Train_encoded_data.npy'
#         # Test_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/Test_encoded_data.npy'
#         # ML_inputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_inputs_train.npy'
#         # ML_outputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_outputs_train.npy'
#         # ML_inputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_inputs_test.npy'
#         # ML_outputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_outputs_test.npy'
        
        
#         # Train_encoded_data= np.load( Train_encoded_data_or,allow_pickle=True)
#         # Test_encoded_data = np.load( Test_encoded_data_or,allow_pickle=True)
#         # ML_inputs_train = np.load( ML_inputs_train_or,allow_pickle=True)
#         # ML_outputs_train = np.load( ML_outputs_train_or,allow_pickle=True)
#         # ML_inputs_test_all = np.load( ML_inputs_test_or,allow_pickle=True)
#         # ML_outputs_test_all= np.load( ML_outputs_test_or,allow_pickle=True)
#         #


#         # F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_RF,'F1_RF',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_RF,'time_RF',natural_scenes_temporal_averaging_directory,str(bin_))
#         # print('F1_RF',F1_RF)
    
#         # F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_nn,'F1_NN',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nn,'time_nn',natural_scenes_temporal_averaging_directory,str(bin_))
#         # print('F1_nn',F1_nn)
    
#         # print("Naive Bayse")
#         # F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_nb,'F1_nb',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nb,'time_nb',natural_scenes_temporal_averaging_directory,str(bin_))
    
#         # print("KNN")
#         # F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_knn,'F1_knn',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_knn,'time_knn',natural_scenes_temporal_averaging_directory,str(bin_))


#         # F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_RF,'F1_RF',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_RF,'time_RF',natural_scenes_temporal_averaging_directory,str(bin_))
        
    
#         # F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_nn,'F1_NN',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nn,'time_nn',natural_scenes_temporal_averaging_directory,str(bin_))
        
    
#         # print("Naive Bayse")
#         # F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_nb,'F1_nb',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nb,'time_nb',natural_scenes_temporal_averaging_directory,str(bin_))
    
#         # print("KNN")
#         # F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_knn,'F1_knn',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_knn,'time_knn',natural_scenes_temporal_averaging_directory,str(bin_))


# # what 

# _______________________________________________________________________________________________________________________

################################################################################## temporal averaging - training model


#     for bin_ in ['0.001','0.002','0.005','0.01','0.025','0.05','0.075','0.075overlap','0.1','0.1overlap','0.125','0.15','0.15overlap','0.2','0.2overlap','0.25'][::-1]:
        
        
# what


# ################################################################################## different brain region


def ML_models_natural_scenes_temporal(session_id):

    print('session_id',session_id)
    session = cache.get_session_data(session_id)
    session_directory = os.path.join('/media/ssd/toktam', str(session_id))    
    session_natural_scenes_directory = os.path.join(session_directory, 'natural_scenes')  
    natural_scenes_temporal_averaging_directory = os.path.join(session_natural_scenes_directory, 'ML_Models_temporal_averaging')

    if not os.path.exists(natural_scenes_temporal_averaging_directory):
        os.mkdir(natural_scenes_temporal_averaging_directory)


    # should be changed based on directory (population level)


    Train_encoded_data_or = '/media/erfan/Elements/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Train_encoded_data.npy'
    Test_encoded_data_or = '/media/erfan/Elements/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Test_encoded_data.npy'
    ML_inputs_train_or = '/media/erfan/Elements/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_inputs_train.npy'
    ML_outputs_train_or = '/media/erfan/Elements/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_outputs_train.npy'
    ML_inputs_test_or = '/media/erfan/Elements/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_inputs_test.npy'
    ML_outputs_test_or = '/media/erfan/Elements/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_outputs_test.npy'
    
    Train_encoded_data_or  = Train_encoded_data_or.replace('757216464', str(session_id))
    Test_encoded_data_or = Test_encoded_data_or.replace('757216464', str(session_id))
    ML_inputs_train_or = ML_inputs_train_or.replace('757216464', str(session_id))
    ML_outputs_train_or = ML_outputs_train_or.replace('757216464', str(session_id))
    ML_inputs_test_or = ML_inputs_test_or.replace('757216464', str(session_id))
    ML_outputs_test_or = ML_outputs_test_or.replace('757216464', str(session_id))

    for bin_ in ['0.001','0.002','0.005','0.01','0.025','0.05','0.125']:
    # ['0.001','0.002','0.005','0.01','0.025','0.05','0.075','0.075overlap','0.1','0.1overlap','0.125','0.15','0.15overlap','0.175','0.175overlap','0.2','0.2overlap','0.225','0.225overlap','0.25'][::-1]:
        
        print("-------------- bin --------------",bin_)
        Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test= get_data_bin(session_id,bin_,Train_encoded_data_or, Test_encoded_data_or , ML_inputs_train_or , ML_outputs_train_or, ML_inputs_test_or, ML_outputs_test_or)
        
            
        Test_encoded_data_all = Test_encoded_data
        ML_inputs_test_all = ML_inputs_test
        ML_outputs_test_all = ML_outputs_test
        
        print("-------------- bin --------------",bin_)
        F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_RF,'F1_RF',natural_scenes_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_RF,'time_RF',natural_scenes_temporal_averaging_directory,str(bin_))
        
        print("-------------- bin --------------",bin_)
        F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_nn,'F1_NN',natural_scenes_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_nn,'time_nn',natural_scenes_temporal_averaging_directory,str(bin_))
        
        print("-------------- bin --------------",bin_)
        print("Naive Bayse")
        F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_nb,'F1_nb',natural_scenes_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_nb,'time_nb',natural_scenes_temporal_averaging_directory,str(bin_))
        
        print("-------------- bin --------------",bin_)
        print("KNN")
        F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_knn,'F1_knn',natural_scenes_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_knn,'time_knn',natural_scenes_temporal_averaging_directory,str(bin_))

def ML_models_natural_scenes_spatial(session_id):

    print('session_id',session_id)
    session = cache.get_session_data(session_id)
    session_directory = os.path.join('/media/ssd/toktam', str(session_id))    
    session_natural_scenes_directory = os.path.join(session_directory, 'natural_scenes')  
    natural_scenes_spatial_averaging_directory = os.path.join(session_natural_scenes_directory, 'ML_Models_spatial_averaging')

    if not os.path.exists(natural_scenes_spatial_averaging_directory):
        os.mkdir(natural_scenes_spatial_averaging_directory)


    # should be changed based on directory (population level)

    Train_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/neuron_level/Train_encoded_data.npy'
    Test_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/neuron_level/Test_encoded_data.npy'
    ML_inputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/neuron_level/ML_inputs_train.npy'
    ML_outputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/neuron_level/ML_outputs_train.npy'
    ML_inputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/neuron_level/ML_inputs_test.npy'
    ML_outputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/neuron_level/ML_outputs_test.npy'
    
    
    Train_encoded_data_or  = Train_encoded_data_or.replace('757216464', str(session_id))
    Test_encoded_data_or = Test_encoded_data_or.replace('757216464', str(session_id))
    ML_inputs_train_or = ML_inputs_train_or.replace('757216464', str(session_id))
    ML_outputs_train_or = ML_outputs_train_or.replace('757216464', str(session_id))
    ML_inputs_test_or = ML_inputs_test_or.replace('757216464', str(session_id))
    ML_outputs_test_or = ML_outputs_test_or.replace('757216464', str(session_id))

    
    for level in ['neuron_level','population_level','each_region_level','super_region_level','whole_brain']:
        

        Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test= get_data_level(session_id,level,Train_encoded_data_or, Test_encoded_data_or , ML_inputs_train_or , ML_outputs_train_or, ML_inputs_test_or, ML_outputs_test_or)
        
            
        Test_encoded_data_all = Test_encoded_data
        ML_inputs_test_all = ML_inputs_test
        ML_outputs_test_all = ML_outputs_test
        
        
        F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_RF,'F1_RF',natural_scenes_spatial_averaging_directory,level)
        save_result_to_directory(time_RF,'time_RF',natural_scenes_spatial_averaging_directory,level)
        
    
        F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_nn,'F1_NN',natural_scenes_spatial_averaging_directory,level)
        save_result_to_directory(time_nn,'time_nn',natural_scenes_spatial_averaging_directory,level)
        
    
        print("Naive Bayse")
        F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_nb,'F1_nb',natural_scenes_spatial_averaging_directory,level)
        save_result_to_directory(time_nb,'time_nb',natural_scenes_spatial_averaging_directory,level)
    
        print("KNN")
        F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_knn,'F1_knn',natural_scenes_spatial_averaging_directory,level)
        save_result_to_directory(time_knn,'time_knn',natural_scenes_spatial_averaging_directory,level)


def multiprocessing_ML_models():
    #,754312389,754829445,732592105 left
    # 763673393 got problem
    #757216464, 757970808, 759883607,761418226,799864342
    with Pool(32) as p:
        session_IDs = [754312389, 732592105, 754829445 ,763673393]
        p.map(ML_models_natural_scenes_spatial,session_IDs )


################################################################################## population level temporal averaging


# br_EX_IN_units_path = 'wild_mice/757216464/gabor/spatial_averaging/population_level/br_EX_IN_units.npy'
# #757216464 , 757970808 , 759883607,761418226,799864342 , 754312389,754829445,732592105 done
# #763673393  error
# for session_id in [ 763673393 ]:

        
#     encoded_all_units = encode_brain_regions_EX_IN(brain_regions)
#     br_EX_IN_units_path_updated = br_EX_IN_units_path.replace('757216464', str(session_id))
#     br_EX_IN_units = np.load(br_EX_IN_units_path_updated,allow_pickle=True)
#     br_EX_IN_units = br_EX_IN_units.item()
#     br_spikes = spikes_seperate_brain_region(brain_regions )
#     # 0.250,0.125,0.050,0.025,0.010,0.005,0.002,
#     #0.250, 0.200, 0.150,0.125, 0.100,0.075,0.050,0.025,0.010,
#     for bin_ in [0.250,0.125,0.050,0.025,0.010,0.005,0.002,0.001]:
        
#         # Train_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/Train_encoded_data.npy'
#         # Test_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/Test_encoded_data.npy'
#         # ML_inputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_inputs_train.npy'
#         # ML_outputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_outputs_train.npy'
#         # ML_inputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_inputs_test.npy'
#         # ML_outputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/ML_outputs_test.npy'
        
        
#         # Train_encoded_data= np.load( Train_encoded_data_or,allow_pickle=True)
#         # Test_encoded_data = np.load( Test_encoded_data_or,allow_pickle=True)
#         # ML_inputs_train = np.load( ML_inputs_train_or,allow_pickle=True)
#         # ML_outputs_train = np.load( ML_outputs_train_or,allow_pickle=True)
#         # ML_inputs_test_all = np.load( ML_inputs_test_or,allow_pickle=True)
#         # ML_outputs_test_all= np.load( ML_outputs_test_or,allow_pickle=True)
#         #


#         # F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_RF,'F1_RF',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_RF,'time_RF',natural_scenes_temporal_averaging_directory,str(bin_))
#         # print('F1_RF',F1_RF)
    
#         # F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_nn,'F1_NN',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nn,'time_nn',natural_scenes_temporal_averaging_directory,str(bin_))
#         # print('F1_nn',F1_nn)
    
#         # print("Naive Bayse")
#         # F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_nb,'F1_nb',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_nb,'time_nb',natural_scenes_temporal_averaging_directory,str(bin_))
    
#         # print("KNN")
#         # F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
#         # save_result_to_directory(F1_knn,'F1_knn',natural_scenes_temporal_averaging_directory,str(bin_))
#         # save_result_to_directory(time_knn,'time_knn',natural_scenes_temporal_averaging_directory,str(bin_))

##################################################################################

print('------------------------------------multiprocessing_ML_models')

with Pool(32) as p:
    session_IDs = [ 763673393]
    p.map(ML_models_natural_scenes_temporal,session_IDs )
    
what 


################################################################################## spatial averaging
print('natural_scenes spatial averaging')


original_str_001 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/F1_HDC.npy'
original_str_002 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.002/F1_HDC.npy'
original_str_005 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.005/F1_HDC.npy'
original_str_010 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.01/F1_HDC.npy'
original_str_025 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.025/F1_HDC.npy'
original_str_050 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.05/F1_HDC.npy'
original_str_075 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.075/F1_HDC.npy'
original_str_075overlap = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.075overlap/F1_HDC.npy'
original_str_100 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.1/F1_HDC.npy'
original_str_100overlap = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.1overlap/F1_HDC.npy'
original_str_125 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.125/F1_HDC.npy'
original_str_150 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.15/F1_HDC.npy'
original_str_150overlap = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.15overlap/F1_HDC.npy'
original_str_200 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.2/F1_HDC.npy'
original_str_200overlap = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.2overlap/F1_HDC.npy'
original_str_250 = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.25/F1_HDC.npy'

def run_ml_model_spatial_averaging_natural_scenes():
    F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
    save_result_to_directory(F1_RF,'F1_RF',natural_scenes_spatial_averaging_directory,averaging_level)
    save_result_to_directory(time_RF,'time_RF',natural_scenes_spatial_averaging_directory,averaging_level)
    print('F1_RF',F1_RF)
    
    F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
    save_result_to_directory(F1_nn,'F1_NN',natural_scenes_spatial_averaging_directory,averaging_level)
    save_result_to_directory(time_nn,'time_nn',natural_scenes_spatial_averaging_directory,averaging_level)
    print('F1_nn',F1_nn)
    
    print("Naive Bayse")
    F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
    save_result_to_directory(F1_nb,'F1_nb',natural_scenes_spatial_averaging_directory,averaging_level)
    save_result_to_directory(time_nb,'time_nb',natural_scenes_spatial_averaging_directory,averaging_level)
    
    print("KNN")
    F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
    save_result_to_directory(F1_knn,'F1_knn',natural_scenes_spatial_averaging_directory,averaging_level)
    save_result_to_directory(time_knn,'time_knn',natural_scenes_spatial_averaging_directory,averaging_level)


br_EX_IN_units_path = 'wild_mice/757216464/gabor/spatial_averaging/population_level/br_EX_IN_units.npy'

# complete  757216464,757970808 , 759883607,761418226,799864342 ,754312389,732592105
# super and whole left 754829445
#757970808 , 761418226 ,763673393 , 759883607,799864342,732592105,754312389,754829445,732592105
for session_id in [754312389,732592105,754829445,763673393]:
    print('session_id',session_id)
    session = cache.get_session_data(session_id)
    session_directory = os.path.join('/media/ssd/toktam', str(session_id))
    if not os.path.exists(session_directory):
        os.mkdir(session_directory)
        
    session_natural_scenes_directory = os.path.join(session_directory, 'natural_scenes')
    if not os.path.exists(session_natural_scenes_directory):
        os.mkdir(session_natural_scenes_directory)
  
    natural_scenes_spatial_averaging_directory = os.path.join(session_natural_scenes_directory, 'ML_Models_spatial_averaging')

    if not os.path.exists(natural_scenes_spatial_averaging_directory):
        os.mkdir(natural_scenes_spatial_averaging_directory)
    
    rf_stim_table = session.stimulus_presentations[session.stimulus_presentations.stimulus_name == 'natural_scenes']  
    rf_stim_table['stimulus_presentation_id'] = rf_stim_table.index.values
    
    scene_presentations = session.get_stimulus_table("natural_scenes")  

    rf_mapping = ReceptiveFieldMapping(session)  
    List_of_frames = list(np.sort(rf_stim_table.frame.unique()))
    List_of_frames.pop(0)
    List_of_frames = list(np.array(List_of_frames).astype(int)) 
    List_of_positions = List_of_frames
    encoder = Encoder(features,D)
    grouped_by_frame = rf_stim_table.groupby('frame')["stimulus_presentation_id"]
    
    brain_regions = whole_brain_regions_check(session.structure_acronyms)
    spikes = spikes_diff_brain_region(brain_regions)
    br_spikes = spikes_seperate_brain_region(brain_regions)
  
    
    encoder = Encoder(features,D)
    brain_regions = whole_brain_regions_check(session.structure_acronyms) #whole brain
    encoded_all_units = encode_all_units_seperating_region(brain_regions)
    unit_IDs = list(spikes.unit_id.unique())
    shuffle_times_train = 4
    shuffle_times_test = 4
    epochs = 3
    
    bin_ = optimal_time_bin(str(session_id),'F1_HDC')
    print('Optimal time bin : ', bin_)
 
    if bin_ in [0.075,0.1,0.150,0.200]:
        crop_overlap_mood = 'overlap'
        time_bins = generate_time_slice_for_overlapping_bin(bin_)
    else:
        crop_overlap_mood = 'crop'
        time_bins = np.linspace(0,0.250- round(0.250-bin_*int(np.floor(0.250/bin_)),3), int(np.floor(0.250/bin_))+1)

    #********************************************************************************************************************
    # neuron level
    print('neuron_level')
    averaging_level = 'neuron_level'
    
    brain_regions = whole_brain_regions_check(session.structure_acronyms)
    encoded_all_units = encode_all_units_seperating_region(brain_regions)
    spikes = spikes_diff_brain_region(brain_regions)
    unit_IDs = list(spikes.unit_id.unique())
    
    
    Train_encoded_data, Test_encoded_data_all , ML_inputs_train ,ML_outputs_train  ,ML_inputs_test_all , ML_outputs_test_all ,Test_To_Shuffle=  New_data_augmentation_shuffling_method(brain_regions,shuffle_times_train,averaging_level)
    Test_encoded_data = Test_encoded_data_all
    ML_inputs_test = ML_inputs_test_all 
    ML_outputs_test = ML_outputs_test_all

    save_all_inputs_to_directory(Train_encoded_data, Test_encoded_data_all , ML_inputs_train ,ML_outputs_train  ,ML_inputs_test_all , ML_outputs_test_all ,  natural_scenes_spatial_averaging_directory,averaging_level)
    
    F1_HDC ,Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all ,time_HDC = Training(epochs,shuffle_times_test,Train_encoded_data, Test_encoded_data,Test_encoded_data_all)
        
    save_result_to_directory(F1_HDC,'F1_HDC',natural_scenes_spatial_averaging_directory,averaging_level)
    save_result_to_directory(time_HDC,'time_HDC',natural_scenes_spatial_averaging_directory,averaging_level)


    save_all_inputs_to_directory_natural_scenes(Train_encoded_data , Test_encoded_data_all , ML_inputs_train , ML_outputs_train , ML_inputs_test_all , ML_outputs_test_all,Test_To_Shuffle, natural_scenes_spatial_averaging_directory,averaging_level)
    
    #********************************************************************************************************************
    # # population level

    
    # #############################
    # # Train_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/Train_encoded_data.npy'
    # # Test_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/Test_encoded_data.npy'
    # # ML_inputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/ML_inputs_train.npy'
    # # ML_outputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/ML_outputs_train.npy'
    # # ML_inputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/ML_inputs_test.npy'
    # # ML_outputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/ML_outputs_test.npy'
    # # # Test_To_Shuffle = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_spatial_averaging/population_level/
    
    # # Train_encoded_data= np.load( Train_encoded_data_or,allow_pickle=True)
    # # Test_encoded_data = np.load( Test_encoded_data_or,allow_pickle=True)
    # # Test_encoded_data_all = Test_encoded_data
    # # ML_inputs_train = np.load( ML_inputs_train_or,allow_pickle=True)
    # # ML_outputs_train = np.load( ML_outputs_train_or,allow_pickle=True)
    # # ML_inputs_test_all = np.load( ML_inputs_test_or,allow_pickle=True)
    # # ML_outputs_test_all= np.load( ML_outputs_test_or,allow_pickle=True)

        
    #********************************************************************************************************************
    # # each-region-level
    # print('each_region_level')
    # averaging_level = 'each_region_level'
    # brain_regions =  whole_brain_regions_check(session.structure_acronyms)
    # encoded_all_units = encode_brain_regions(brain_regions)
    # unit_IDs = brain_regions


    # #********************************************************************************************************************
    # # whole_brain
    

    # #****************
    
    # # save_inputs_to_directory(F1_HDC_whole_brain , natural_scenes_spatial_averaging_directory ,averaging_level)
    
    
