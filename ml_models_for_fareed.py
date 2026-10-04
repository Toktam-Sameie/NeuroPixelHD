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
    Multiprocessing,
    Threshhoding_receptive_field,
    check_in_region,
    check_similarity_to_all_classes,
    encode_decode_position,
    eucli_dis,
    euclidean_distance,
    generate_random_vecotor,
    get_unitID_in_bin,
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


def random_forest(n,ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test,Train_target,Test_target):
    print(" Training Random Forest")
    tic = time.time()
    rfc = RandomForestClassifier(n_estimators=n)
    rfc.fit(ML_inputs_train, Train_target)
    
    
    y_pred_test_RF = rfc.predict(ML_inputs_test)
    
    y_pred_pos_test = label_to_tuple_pos(y_pred_test_RF)
    ML_outputs_test_tuple = [tuple(row) for row in ML_outputs_test]
    euc_dis_test_RF = eucli_dis(ML_outputs_test_tuple,y_pred_pos_test)
    tac = time.time()
    print('time : ',tac - tic)
    print('mean of euclidean distance test RF : ',np.mean(euc_dis_test_RF))
    
    return euc_dis_test_RF , tac - tic
    
def svm(kr,c):
    from sklearn import svm
    print(" Training SVM")
    tic = time.time()
    svmc = svm.SVC(kernel=kr, C=c, decision_function_shape='ovr')
    svmc.fit(ML_inputs_train, list(Train_target))
    print('done fitting with svm')

    y_pred_test_svm = svmc.predict(ML_inputs_test)
    

    y_pred_pos_test = label_to_tuple_pos(y_pred_test_svm)
    ML_outputs_test_tuple = [tuple(row) for row in ML_outputs_test]
    euc_dis_test_svm = eucli_dis(ML_outputs_test_tuple,y_pred_pos_test)
    tac = time.time()
    print('time : ',tac - tic)
    print('mean of euclidean distance test svm : ',np.mean(euc_dis_test_svm))

    return euc_dis_test_svm , tac - tic


def neural_network(hls,ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test,Train_target,Test_target):
    from sklearn.neural_network import MLPClassifier

    print("Training Neural Network")
    tic = time.time()

    # Initialize the neural network classifier
    nn_classifier = MLPClassifier(hidden_layer_sizes=(hls,), activation='relu', solver='adam', random_state=42)

    # Scale the training data
    scaler = StandardScaler()
    ML_inputs_train_normalized = scaler.fit_transform(ML_inputs_train)

    # Fit the neural network on the scaled training data
    nn_classifier.fit(ML_inputs_train_normalized, Train_target)
    print('Done fitting with the neural network')

    # Scale the testing data
    ML_inputs_test_normalized = scaler.transform(ML_inputs_test)

    # Make predictions on the scaled testing data
    y_pred_test_nn = nn_classifier.predict(ML_inputs_test_normalized)

    # Convert predictions and ground truth to tuples for Euclidean distance calculation
    y_pred_pos_test = label_to_tuple_pos(y_pred_test_nn)
    ML_outputs_test_tuple = [tuple(row) for row in ML_outputs_test]

    # Calculate Euclidean distance
    euc_dis_test_nn = eucli_dis(ML_outputs_test_tuple, y_pred_pos_test)

    tac = time.time()
    print('Time:', tac - tic)
    print('Mean of Euclidean distance test neural network:', np.mean(euc_dis_test_nn))

    return euc_dis_test_nn, tac - tic

# Assuming ML_inputs_train, ML_inputs_test, Train_target, ML_outputs_test, label_to_tuple_pos, eucli_dis are defined elsewhere


def naive_bayes(ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test,Train_target,Test_target):
    print("Training Naive Bayes")
    tic = time.time()

    nb_classifier = GaussianNB()

    nb_classifier.fit(ML_inputs_train, Train_target)
    print('Done fitting with Naive Bayes')


    # Predict using the trained Naive Bayes classifier
    # y_pred_train_nb = nb_classifier.predict(ML_inputs_train)
    y_pred_test_nb = nb_classifier.predict(ML_inputs_test)

    # Convert numerical labels back to tuple format
    # y_pred_pos_train = label_to_tuple_pos(y_pred_train_nb)
    y_pred_pos_test = label_to_tuple_pos(y_pred_test_nb)

    ML_outputs_test_tuple = [tuple(row) for row in ML_outputs_test]
    euc_dis_test_nb = eucli_dis(ML_outputs_test_tuple, y_pred_pos_test)
    tac = time.time()
    print('Time:', tac - tic)
    print('Mean of Euclidean distance test Naive Bayes:', np.mean(euc_dis_test_nb))

    return euc_dis_test_nb, tac - tic

# Function for training and evaluating KNN
def knn(ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test,Train_target,Test_target,k):
    print("Training KNN")
    tic = time.time()

    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    knn_classifier.fit(ML_inputs_train, Train_target)
    print(f'Done fitting with KNN (k={k})')


    # Predict using the trained KNN classifier
    # y_pred_train_knn = knn_classifier.predict(ML_inputs_train)
    y_pred_test_knn = knn_classifier.predict(ML_inputs_test)

    # Convert numerical labels back to tuple format
    # y_pred_pos_train = label_to_tuple_pos(y_pred_train_knn)
    y_pred_pos_test = label_to_tuple_pos(y_pred_test_knn)

    ML_outputs_test_tuple = [tuple(row) for row in ML_outputs_test]
    euc_dis_test_knn = eucli_dis(ML_outputs_test_tuple, y_pred_pos_test)

    tac = time.time()
    print('Time:', tac - tic)
    print(f'Mean of Euclidean distance test KNN (k={k}):', np.mean(euc_dis_test_knn))

    return euc_dis_test_knn, tac - tic

# Function for training and evaluating Gradient Boosting
def gradient_boosting():
    print("Training Gradient Boosting")
    tic = time.time()

    gb_classifier = GradientBoostingClassifier()

    gb_classifier.fit(ML_inputs_train, Train_target)
    print('Done fitting with Gradient Boosting')

    tac = time.time()
    print('Time:', tac - tic)

    # Predict using the trained Gradient Boosting classifier
    y_pred_train_gb = gb_classifier.predict(ML_inputs_train)
    y_pred_test_gb = gb_classifier.predict(ML_inputs_test)

    # Convert numerical labels back to tuple format
    y_pred_pos_train = label_to_tuple_pos(y_pred_train_gb)
    y_pred_pos_test = label_to_tuple_pos(y_pred_test_gb)

    ML_outputs_test_tuple = [tuple(row) for row in ML_outputs_test]
    euc_dis_test_gb = eucli_dis(ML_outputs_test_tuple, y_pred_pos_test)

    print('Mean of Euclidean distance test Gradient Boosting:', np.mean(euc_dis_test_gb))

    return euc_dis_test_gb, tac - tic


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

def Adaptive_training_gabor_efficient(Train_data,Test_data):
    
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


##################################################################################

def get_data_bin(session_id,bin_):
    
    Train_encoded_data= np.load( Train_encoded_data_or.replace('0.001', bin_),allow_pickle=True)
    Test_encoded_data = np.load( Test_encoded_data_or.replace('0.001', bin_),allow_pickle=True)
    ML_inputs_train = np.load( ML_inputs_train_or.replace('0.001', bin_),allow_pickle=True)
    ML_outputs_train = np.load( ML_outputs_train_or.replace('0.001', bin_),allow_pickle=True)
    ML_inputs_test = np.load( ML_inputs_test_or.replace('0.001', bin_),allow_pickle=True)
    ML_outputs_test= np.load( ML_outputs_test_or.replace('0.001', bin_),allow_pickle=True)
    return Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test

    
################################################################################## training ML Models
print('training ML Models')
for session_id in [757216464]:
    print('session_id',session_id)
    session = cache.get_session_data(session_id)
    session_directory = os.path.join('/media/ssd/toktam', str(session_id))    
    session_gabor_directory = os.path.join(session_directory, 'gabor')  
    gabor_temporal_averaging_directory = os.path.join(session_gabor_directory, 'ML_Models_temporal_averaging')

    if not os.path.exists(gabor_temporal_averaging_directory):
        os.mkdir(gabor_temporal_averaging_directory)
    
    rf_mapping = ReceptiveFieldMapping(session)  
    rf_stim_table = session.stimulus_presentations[session.stimulus_presentations.stimulus_name == 'gabors']  
    rf_stim_table['stimulus_presentation_id'] = rf_stim_table.index.values
    rf_stim_table['Position'] = list(zip(rf_stim_table['x_position'], rf_stim_table['y_position']))
    rf_stim_table['Position'] = [((rf_stim_table['Position'].iloc[i][0]+40)/10, (rf_stim_table['Position'].iloc[i][1]+40)/10) for  i in range(len(rf_stim_table['Position']))]
    List_of_positions = list(np.sort(rf_stim_table.Position.unique()))
    grouped_by_Position = rf_stim_table.groupby('Position')["stimulus_presentation_id"]
    shuffle_times = 10
    epochs = 2
    scene_presentations = session.get_stimulus_table("gabors")  
    whole_brain = list(session.units["ecephys_structure_acronym"].unique())
    
    
    for bin_ in ['0.001','0.002','0.005','0.01','0.025','0.05','0.075','0.075overlap','0.1','0.1overlap','0.125','0.15','0.15overlap','0.2','0.2overlap','0.25'][::-1]:
        # 
        print("bin",bin_)
        Train_encoded_data, Test_encoded_data , ML_inputs_train , ML_outputs_train, ML_inputs_test, ML_outputs_test= get_data_bin(bin_)
        encode_pos , decode_pos = encode_decode_position(List_of_positions)
        brain_regions = whole_brain_regions_check(session.structure_acronyms) #whole brain
        
        
        Train_target = target_pos_to_num( ML_outputs_train,encode_pos)
        Test_target = target_pos_to_num( ML_outputs_test ,encode_pos)
        
        euc_dis_test_HDC , time_HDC =  Adaptive_training_data_augmentation_shuffling(Original_Train_encoded_data,Original_Test_encoded_data,brain_regions,List_of_positions,epochs,shuffle_times)

        save_result_to_directory(euc_dis_test_HDC,'euc_dis_HDC',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_HDC,'time_HDC',gabor_temporal_averaging_directory,str(bin_))
        
        euc_dis_test_RF , time_RF = random_forest(10)
        save_result_to_directory(euc_dis_test_RF,'euc_dis_RF',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_RF,'time_RF',gabor_temporal_averaging_directory,str(bin_))
        
        euc_dis_test_svm , time_svm = svm('rbf',0.5)
        save_result_to_directory(euc_dis_test_svm,'euc_dis_svm',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_svm,'time_svm',gabor_temporal_averaging_directory,str(bin_))
    
        euc_dis_test_nn , time_nn = neural_network(10)
        save_result_to_directory(euc_dis_test_nn,'euc_dis_NN',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_nn,'time_nn',gabor_temporal_averaging_directory,str(bin_))
        
        print("LDA")
        euc_dis_test_lda, time_lda = lda_algorithm()
        save_result_to_directory(euc_dis_test_lda,'euc_dis_lda',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_lda,'time_lda',gabor_temporal_averaging_directory,str(bin_))
    
        print("Logistic Regression")
        euc_dis_test_lr, time_lr = logistic_regression_algorithm()
        save_result_to_directory(euc_dis_test_lr,'euc_dis_lr',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_lr,'time_lr',gabor_temporal_averaging_directory,str(bin_))
    
        print("Naive Bayse")
        euc_dis_test_nb, time_nb = naive_bayes()
        save_result_to_directory(euc_dis_test_nb,'euc_dis_nb',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_nb,'time_nb',gabor_temporal_averaging_directory,str(bin_))
    
        print("KNN")
        euc_dis_test_knn, time_knn = knn(k=18)
        save_result_to_directory(euc_dis_test_knn,'euc_dis_knn',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_knn,'time_knn',gabor_temporal_averaging_directory,str(bin_))
    
        print("gradient_boosting")
        euc_dis_test_gb, time_gb = gradient_boosting()
        save_result_to_directory(euc_dis_test_gb,'euc_dis_gb',gabor_temporal_averaging_directory,str(bin_))
        save_result_to_directory(time_gb,'time_gb',gabor_temporal_averaging_directory,str(bin_))
        
