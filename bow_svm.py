import os, time, pdb, math
import numpy as np
from numpy import linalg as LA
import deepdish as dd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def normalize_feat(x):
    '''
        X: n*p, where n is # of samples, p is feat dim
        Here, only L2 is used to normalize feat,
        other normalized method should also be considered
    '''
    norm = LA.norm(x, ord = 2, axis = 1)
    return x / norm[:, np.newaxis]

def generate_kmeans_model(category_folder_name):
    # 1. read files about slices
    folder = 'train'
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_googLeNet.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_googLeNet.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features
    print('Begining Slice BOW...')
    # clustering features
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
    kmeans_model.fit(feats)

    return kmeans_model

def generate_patient_bow_feats(folder, category_folder_name, kmeans_model):
    '''
        1. read files    2. produce slice bow features
        3. produce patient bow feature of all the timepoints
    '''
    # 1. read files about slices
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_googLeNet.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_googLeNet.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features
    # print('Begining Slice BOW...')
    # predict the label that X belongs to which index of centers
    slices_bow_feats = kmeans_model.predict(feats)

    # 3. produce patient bow feature
    # print('Beginning Patient BOW Feat...')
    patient_bow_list = []
    patient_label_gt_list = []
    patients_bow = {}
    patient_label_gt = {}
    ## Get the true and predicted labels of subjects
    for i, filename in enumerate(filenames):
        # print('{} slice'.format(filename))
        patient_name = filename[2:17]
        if patient_name[:11] == 'ADNI_bet-B_':
            patient_name = 'ADNI_bet-B_'
        bow_feat = slices_bow_feats[i]
        if patient_name in patients_bow.keys(): # calculate the probability of the same subjects
            patients_bow[patient_name][bow_feat] += 1.0
        else:
            patient_label_gt[patient_name] = int(filename[0])
            patient_bow = np.zeros(num_clusters)
            patient_bow[bow_feat] += 1.0
            patients_bow[patient_name] = patient_bow

    for pid in patients_bow.keys():
        # pdb.set_trace()
        patient_label_gt_list.append(patient_label_gt[pid])
        patient_bow_list.append(patients_bow[pid])

    patient_bow_list = normalize_feat(patient_bow_list)   #normalizing do great help to results

    return patient_bow_list, patient_label_gt_list

def evaluate_svm_classifier(svm_model, patients_bow_feats, patients_label_gt):
    # pdb.set_trace()
    # score = svm_model.score(patients_bow_feats, patient_label_gt)
    # print('mean_acc: {}'.format(score))
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    labels_pred = svm_model.predict(patients_bow_feats)
    pdb.set_trace()
    for i, patient_label_pred in enumerate(labels_pred):
        # patient_label_pred = pid
        patient_label_gt = patients_label_gt[i]
        if patient_label_pred == 0:
            PP += 1.0
            if patient_label_gt == patient_label_pred:
                TP += 1.0
            else:
                FP += 1.0
        else:
            PN += 1.0
            if patient_label_gt == patient_label_pred:
                TN += 1.0
            else:
                FN += 1.0
    PPV = 1.0*TP/PP
    FOR = 1.0*FN/PN
    FDR = 1.0*FP/PP
    NPV = 1.0*TN/PN
    patient_acc = 1.0*(TP+TN)/len(patients_bow_feats)

    print('Patient_acc {}'.format(patient_acc))
    print('Confusion matrix:\n{}  {}  \n{}  {}'.format(PPV, FOR, FDR, NPV))
    num_patients_0 = np.sum(np.array(patients_label_gt.values()) == 0)
    num_patients_1 = np.sum(np.array(patients_label_gt.values()) == 1)
    print('GT: {} patients {} MCI_patients {} NC_patients'.format(len(patients_label_gt), num_patients_0, num_patients_1))


    # score_compute =  np.sum(pre == patient_label_gt)
    # print('Train_mean_acc_compute: {}'.format((1.0*score_compute/len(patients_bow_feats))))

    return patient_acc

def train_model_help(X_train, y_train, model_strs):
    '''
        Does not work now
    '''
    '''
        select different models
    '''
    if 'logit' in model_strs:
        print '---------- grid search for logistic regression -------'
        est = LogisticRegression(n_jobs=-1)
        tuned_parameters = [{'penalty':['l1', 'l2'], 'C':[0.01, 0.1, 1]}]

    if 'svm' in model_strs:
        print '-------- grid search for svm --------'
        est = svm.SVC()
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10]}]

    if 'nn' in model_strs:
        print '------- grid search for nearest neighbor classifier -------'
        est = KNeighborsClassifier(n_jobs=-1)
        tuned_parameters = [{'n_neighbors': [5,10,20]}]

    model = GridSearchCV(est, tuned_parameters, cv=5)
    model.fit(X_train, y_train)
    return model

def main_funct(num_clusters, category_folder_name):
    '''
        1. generate kmeans_model  2. generate SVM model based on patients' bow feats of train data
        3. evaluate svm_model based on the bow feats of train, vald and test data
    '''

    # 1. generate kmeans_model
    kmeans_model = generate_kmeans_model(category_folder_name)
    # 2. extract patients' BOW feats of train data and train svm model
    print '-------------Train and evaluate SVM model-------------'
    folder = 'train'
    patients_bow_feats, patient_label_gt = generate_patient_bow_feats(folder, category_folder_name, kmeans_model)
    est = svm.SVC()
    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'C': [0.001, 0.01, 1, 10]}]
    svm_model = GridSearchCV(est, tuned_parameters, cv=5)  #grid search
    svm_model.fit(patients_bow_feats, patient_label_gt)
    # 3. evaluate SVM model based on the bow feats of training data
    train_score = evaluate_svm_classifier(svm_model, patients_bow_feats, patient_label_gt)
    print '-------------Evaluate Test Data-------------'
    folder = 'test'
    patients_bow_feats, patient_label_gt = generate_patient_bow_feats(folder, category_folder_name, kmeans_model)
    test_score = evaluate_svm_classifier(svm_model, patients_bow_feats, patient_label_gt)
    print '-------------Evaluate Vald Data-------------'
    folder = 'vald'
    patients_bow_feats, patient_label_gt = generate_patient_bow_feats(folder, category_folder_name, kmeans_model)
    vald_score = evaluate_svm_classifier(svm_model, patients_bow_feats, patient_label_gt)

    return train_score, test_score, vald_score

if __name__ == '__main__':
    num_clusters = 32
    category_folder_name = 'NCvsAD'
    # category_folder_name = 'NCvsMCI'
    suffix = '_normslice'
    folder_name = category_folder_name+suffix
    # score_max = []  # classification acc under diff 'num_clusters'
    # for num_clusters in xrange(3, 15, 1):
    print('================= {} Num_Cluster {} ================='.format(category_folder_name, num_clusters))
    train_score, test_score, vald_score = main_funct(num_clusters, folder_name)
    # score_max.append([train_score, test_score, vald_score])
    # print('train, test, vald')
    # for score in score_max:
    #     print(score)
    # pdb.set_trace()
    # print('Over')
