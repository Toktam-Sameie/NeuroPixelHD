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
    get_unitID_in_bin,
    give_max_frame_randomly,
    give_max_position_randomly,
    interpolate_time_hypervectors_window,
    make_HV_of_units_in_binـillumination_method,
    make_accuracy_dict,
    make_classes,
    plot_heatmap_of_region_accuracy,
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
eligible_brain_regions = ['VISp','VISam', 'VISal', 'VISrl', 'VISmma','VISpm', 'VISl','CA1', 'CA2', 'CA3','SUB', 'ProS','DG','TH', 'LP', 'LGv', 'LGd', 'PP', 'PIL', 'MGv', 'PO', 'Eth', 'POL']
#,'APN','ZI'
# , 'VPL', 'IGL',' SGN', 'VPN', 'LD', 'POT','MRN', 'MB', 'NOT','PPT','OP','SCig'


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
    Classes = np.zeros((len(List_of_frames),D))
    time_all = []
    for epoch in tqdm(range(epochs)):
        tic = time.time()
        

        # Train HDC Model
        for i in range(len(Train_encoded_data)):
            frame = Train_encoded_data[i][0]
            data = np.array(Train_encoded_data[i][1])


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
    rearrenged_data = np.transpose(np.stack([row[1] for row in data]))
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
    print('encoded_data_stim',len(encoded_data_stim))
    if frame == predicted_frame:
        Classes[frame] += eta_center*(1 - deltas[frame])*encoded_data_stim
        
    else:
        print('deltas[predicted_frame]',deltas[predicted_frame])
        print('deltas[frame]',deltas[frame])
        print('eta_center',eta_center)
        Classes[predicted_frame] -= eta_center*(deltas[predicted_frame] - deltas[frame])*encoded_data_stim
        Classes[frame] += eta_center*encoded_data_stim
        
    return Classes


def Training_(epochs,shuffle_times_test,Train_encoded_data,Test_encoded_data,Test_encoded_data_all ):  
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


################################################################################## temporal averaging
print('Naural scenes -> temporal averaging') # incomplete
#759883607 : till 0.05 I guess it run
#757970808 , 759883607, not complete , 763673393 ,(not at all)
# complete 761418226,799864342,
# # #757216464 , 757970808 , 759883607,761418226,799864342,763673393 ,754312389,754829445,732592105
# 754312389,754829445,732592105 left whole
for session_id in [ 757216464,757970808]:

    print('session_id',session_id)
    session = cache.get_session_data(session_id)
    session_directory = os.path.join('/media/ssd/toktam', str(session_id))
    if not os.path.exists(session_directory):
        os.mkdir(session_directory)
        
    session_natural_scenes_directory = os.path.join(session_directory, 'natural_scenes')
    if not os.path.exists(session_natural_scenes_directory):
        os.mkdir(session_natural_scenes_directory)
  
    natural_scenes_temporal_averaging_directory = os.path.join(session_natural_scenes_directory, 'ML_Models_temporal_averaging')

    if not os.path.exists(natural_scenes_temporal_averaging_directory):
        os.mkdir(natural_scenes_temporal_averaging_directory)
    
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


    Train_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Train_encoded_data.npy'
    Test_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Test_encoded_data.npy'
    ML_inputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_inputs_train.npy'
    ML_outputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_outputs_train.npy'
    ML_inputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_inputs_test.npy'
    ML_outputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_outputs_test.npy'
    Test_To_Shuffle_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Test_To_Shuffle.npy'
    
    
    Train_encoded_data_or  = Train_encoded_data_or.replace('757216464', str(session_id))
    Test_encoded_data_or = Test_encoded_data_or.replace('757216464', str(session_id))
    ML_inputs_train_or = ML_inputs_train_or.replace('757216464', str(session_id))
    ML_outputs_train_or = ML_outputs_train_or.replace('757216464', str(session_id))
    ML_inputs_test_or = ML_inputs_test_or.replace('757216464', str(session_id))
    ML_outputs_test_or = ML_outputs_test_or.replace('757216464', str(session_id))
    Test_To_Shuffle_or = Test_To_Shuffle_or.replace('757216464', str(session_id))
    
    for bin_ in [0.200, 0.150, 0.100,0.075]:
        crop_overlap_mood = 'crop'
        print('crop     :','   bin_',bin_)
        time_bins = np.linspace(0,0.250- round(0.250-bin_*int(np.floor(0.250/bin_)),3), int(np.floor(0.250/bin_))+1)

        
        Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test,Test_To_Shuffle= get_data_bin(session_id,str(bin_),Train_encoded_data_or, Test_encoded_data_or , ML_inputs_train_or , ML_outputs_train_or, ML_inputs_test_or, ML_outputs_test_or,Test_To_Shuffle_or)
        
        Test_encoded_data = list(Test_encoded_data)    
        Test_encoded_data_all = Test_encoded_data
        ML_inputs_test_all = list(ML_inputs_test)
        ML_outputs_test_all = list(ML_outputs_test)
        ML_inputs_test = list(ML_inputs_test)
        ML_outputs_test = list(ML_outputs_test)

            
        F1_HDC ,Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all ,time_HDC = Training(epochs,shuffle_times_test,Train_encoded_data, Test_encoded_data,Test_encoded_data_all)
        
        save_result_to_directory(F1_HDC,'F1_HDC',natural_scenes_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_HDC,'time_HDC',natural_scenes_temporal_averaging_directory,str(bin_))


    # overlapping structure
    # for bin_ in [0.075, 0.100, 0.150, 0.200 ]:
    #     crop_overlap_mood = 'overlap'
    #     print('overlap     :','     bin_',bin_)
    #     time_bins = generate_time_slice_for_overlapping_bin(bin_)
        
    
what 
################################################################################## temporal averaging - training model


#     for bin_ in ['0.001','0.002','0.005','0.01','0.025','0.05','0.075','0.075overlap','0.1','0.1overlap','0.125','0.15','0.15overlap','0.2','0.2overlap','0.25'][::-1]:
        
        
# what


# ################################################################################## different brain region


def ML_models_natural_scenes(session_id):

    print('session_id',session_id)
    session = cache.get_session_data(session_id)
    session_directory = os.path.join('/media/ssd/toktam', str(session_id))    
    session_natural_scenes_directory = os.path.join(session_directory, 'natural_scenes')  
    natural_scenes_temporal_averaging_directory = os.path.join(session_natural_scenes_directory, 'ML_Models_temporal_averaging')

    if not os.path.exists(natural_scenes_temporal_averaging_directory):
        os.mkdir(natural_scenes_temporal_averaging_directory)


    # should be changed

    Train_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Train_encoded_data.npy'
    Test_encoded_data_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Test_encoded_data.npy'
    ML_inputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_inputs_train.npy'
    ML_outputs_train_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_outputs_train.npy'
    ML_inputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_inputs_test.npy'
    ML_outputs_test_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/ML_outputs_test.npy'
    Test_To_Shuffle_or = '/media/ssd/toktam/757216464/natural_scenes/ML_Models_temporal_averaging/0.001/Test_To_Shuffle.npy'
    
    
    Train_encoded_data_or  = Train_encoded_data_or.replace('757216464', str(session_id))
    Test_encoded_data_or = Test_encoded_data_or.replace('757216464', str(session_id))
    ML_inputs_train_or = ML_inputs_train_or.replace('757216464', str(session_id))
    ML_outputs_train_or = ML_outputs_train_or.replace('757216464', str(session_id))
    ML_inputs_test_or = ML_inputs_test_or.replace('757216464', str(session_id))
    ML_outputs_test_or = ML_outputs_test_or.replace('757216464', str(session_id))
    Test_To_Shuffle_or = Test_To_Shuffle_or.replace('757216464', str(session_id))
    
    # for bin_ in ['0.001','0.002','0.005','0.01','0.025','0.05','0.075','0.075overlap','0.1','0.1overlap','0.125','0.15','0.15overlap','0.2','0.2overlap','0.25'][::-1]:
    for bin_ in ['0.075','0.1','0.15','0.2'][::-1]: 
        print("bin",bin_)
 
        if bin_ in ['0.075overlap','0.1overlap','0.15overlap','0.2overlap']:
            crop_overlap_mood = 'overlap'
            
        else:
            crop_overlap_mood = 'crop'
            

        Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test,Test_To_Shuffle= get_data_bin(session_id,bin_,Train_encoded_data_or, Test_encoded_data_or , ML_inputs_train_or , ML_outputs_train_or, ML_inputs_test_or, ML_outputs_test_or,Test_To_Shuffle_or)
        
            
        Test_encoded_data_all = Test_encoded_data
        ML_inputs_test_all = ML_inputs_test
        ML_outputs_test_all = ML_outputs_test
    
        F1_HDC ,Test_encoded_data_all , ML_inputs_test_all , ML_outputs_test_all ,time_HDC = Training(epochs,shuffle_times_test,Train_encoded_data, Test_encoded_data,Test_encoded_data_all,Test_To_Shuffle,crop_overlap_mood)
        
        save_result_to_directory(F1_HDC,'F1_HDC',natural_scenes_temporal_averaging_directory,bin_)
        save_result_to_directory(time_HDC,'time_HDC',natural_scenes_temporal_averaging_directory,bin_)
        
        F1_RF , time_RF = random_forest(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_RF,'F1_RF',natural_scenes_temporal_averaging_directory,bin_)
        save_result_to_directory(time_RF,'time_RF',natural_scenes_temporal_averaging_directory,bin_)
        
    
        F1_nn , time_nn = neural_network(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_nn,'F1_NN',natural_scenes_temporal_averaging_directory,bin_)
        save_result_to_directory(time_nn,'time_NN',natural_scenes_temporal_averaging_directory,bin_)
        
    
        print("Naive Bayse")
        F1_nb, time_nb = naive_bayes(ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_nb,'F1_nb',natural_scenes_temporal_averaging_directory,bin_)
        save_result_to_directory(time_nb,'time_nb',natural_scenes_temporal_averaging_directory,bin_)
    
        print("KNN")
        F1_knn, time_knn = knn(5,ML_inputs_train , ML_outputs_train,ML_inputs_test_all,ML_outputs_test_all)
        save_result_to_directory(F1_knn,'F1_knn',natural_scenes_temporal_averaging_directory,bin_)
        save_result_to_directory(time_knn,'time_knn',natural_scenes_temporal_averaging_directory,bin_)

session_id = 757216464
print('session_id',session_id)
session = cache.get_session_data(session_id)

rf_stim_table = session.stimulus_presentations[session.stimulus_presentations.stimulus_name == 'natural_scenes']  
rf_stim_table['stimulus_presentation_id'] = rf_stim_table.index.values

scene_presentations = session.get_stimulus_table("natural_scenes")  

rf_mapping = ReceptiveFieldMapping(session)  
List_of_frames = list(np.sort(rf_stim_table.frame.unique()))
List_of_frames.pop(0)
List_of_frames = list(np.array(List_of_frames).astype(int)) 
List_of_positions = List_of_frames

shuffle_times_train = 4
shuffle_times_test = 4
epochs = 3
def multiprocessing_ML_models():
    #
    with Pool(32) as p:
        session_IDs = [757216464,757970808]
        p.map(ML_models_natural_scenes,session_IDs )

print('------------------------------------multiprocessing_ML_models------------------------------------')
multiprocessing_ML_models()


