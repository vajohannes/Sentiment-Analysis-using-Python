import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy import stats


class KNNModel:
  def __init__(self, k, p):
    self.k = k
    self.p = p

  def fit(self, train_dataset, train_labels):
    self.train_dataset = train_dataset
    self.train_labels = train_labels

  def compute_Minkowski_distance(self, test_dataset):
    temp1 = np.expand_dims(test_dataset, axis=1) 
    temp2 = np.abs(self.train_dataset - temp1) ** self.p 
    temp3 = np.sum(temp2, axis=2, keepdims=False) 
    result = temp3 ** (1 / self.p) 
    return result

  def find_k_nearest_neighbor_labels(self, test_dataset):
    return np.take(self.train_labels[:, np.newaxis], np.argsort(self.compute_Minkowski_distance(test_dataset), axis=1)[:, :self.k], mode="wrap")

  def predict(self, test_dataset):
    neighbor_labels = self.find_k_nearest_neighbor_labels(test_dataset)
    predict_mode = stats.mode(neighbor_labels, axis=1, keepdims=False)
    return predict_mode.mode.flatten()


def generate_confusion_matrix(test_predict, test_labels):
  num_class = int(np.amax(test_labels) + 1)
  confusion_matrix = np.zeros((num_class, num_class))
  for i in range(0, num_class):
    for j in range(0, num_class):
      confusion_matrix[i, j] = ((test_labels == i) & (test_predict == j)).sum()  
  return confusion_matrix


def calculate_accuracy(test_predict, test_labels):
  confusion_matrix = generate_confusion_matrix(test_predict, test_labels)
  return np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix)


def calculate_precision(test_predict, test_labels):
  confusion_matrix = generate_confusion_matrix(test_predict, test_labels)
  precision = confusion_matrix.diagonal() / np.sum(confusion_matrix, axis=0)
  # precision = np.nan_to_num(precision, nan=1.0)
  return np.average(precision)


def calculate_recall(test_predict, test_labels):
  confusion_matrix = generate_confusion_matrix(test_predict, test_labels)
  recall = confusion_matrix.diagonal() / np.sum(confusion_matrix, axis=1)
  # recall = np.nan_to_num(recall, nan=1.0)
  return np.average(recall)


def calculate_macro_f1(test_predict, test_labels):
  p = calculate_precision(test_predict, test_labels)
  r = calculate_recall(test_predict, test_labels)
  return 2 * p * r / (p + r)


def calculate_MCC_score(test_predict, test_labels):
  confusion_matrix = generate_confusion_matrix(test_predict, test_labels)
  c = np.sum(confusion_matrix.diagonal())
  s = np.sum(confusion_matrix)
  p = np.sum(confusion_matrix, axis=0)
  t = np.sum(confusion_matrix, axis=1)
  MCC = (c * s - np.sum(p * t)) / np.sqrt((s ** 2 - np.sum(p ** 2)) * (s ** 2 - np.sum(t ** 2)))
  return MCC
  #return np.nan_to_num(MCC, nan=0.0)


class DFoldCV:
  def __init__(self, X, y, k_list, p_list, d, eval_metric):
    self.X = X
    self.y = y
    self.k_list = k_list
    self.p_list = p_list
    self.d = d
    self.eval_metric = eval_metric

  def generate_folds(self):
    dataset = np.concatenate((self.X, self.y[:, np.newaxis]), axis=1)
    dataset_d_folds = np.array_split(dataset, self.d, axis=0)
    train_d_folds = [np.concatenate((dataset_d_folds[0:fold] + dataset_d_folds[fold+1:self.d]), axis=0) for fold in range(self.d)] # Python list "+" operator performs list concatenation.
    #test_d_folds = [dataset_d_folds[fold] for fold in range(self.d)]
    return train_d_folds, dataset_d_folds #test_d_folds
  
  def cross_validate(self):
    train_d_folds, test_d_folds = self.generate_folds()
    scores = np.zeros((len(self.k_list), len(self.p_list), self.d))

    for k_index, k in enumerate(self.k_list):
      for p_index, p in enumerate(self.p_list):
        knn_model = KNNModel(k, p)
        for fold in range(self.d):
          X_train, y_train = (train_d_folds[fold])[:, :-1], (train_d_folds[fold])[:, -1]
          X_test, y_test = (test_d_folds[fold])[:, :-1], (test_d_folds[fold])[:, -1]
          knn_model.fit(X_train, y_train)
          scores[k_index, p_index, fold] = self.eval_metric(knn_model.predict(X_test), y_test)
    return scores

  def validate_best_parameters(self):
    scores = self.cross_validate()
    scores_overall = np.average(scores, axis=2)
    parameter_index = np.unravel_index(np.argmax(scores_overall, axis=None), scores_overall.shape)
    k_best = self.k_list[parameter_index[0]]
    p_best = self.p_list[parameter_index[1]]
    return k_best, p_best


if __name__ == '__main__':
  train_dataset = sparse.load_npz("train_dataset.npz")
  test_dataset = sparse.load_npz("test_dataset.npz")
  train_dataset = train_dataset.toarray()
  test_dataset = test_dataset.toarray()
  train_labels = np.load("train_labels.npy")
  test_labels = np.load("test_labels.npy")

  knn_model = KNNModel(10, 2)
  knn_model.fit(train_dataset, train_labels)
  dist = knn_model.compute_Minkowski_distance(test_dataset)
  print(f"The Minkowski distance between the first five test samples and the first five training samples are:\n {dist[ : 5, : 5]}") # should be [[1.40488545 1.41421356 1.40473647 1.41421356 1.40205505]
                                                                                                                                    #            [1.40172965 1.41421356 1.40153004 1.41421356 1.39793611]
                                                                                                                                    #            [1.40573171 1.41421356 1.40559629 1.41421356 1.40315911]
                                                                                                                                    #            [1.40403747 1.41421356 1.40387491 1.41421356 1.40094856]
                                                                                                                                    #            [1.41421356 1.39611886 1.41421356 1.39841935 1.41421356]]
  k_nearest_neighbor_labels = knn_model.find_k_nearest_neighbor_labels(test_dataset)
  print(f"The k nearest neighbor labels for the first five test samples are:\n {k_nearest_neighbor_labels[ : 5, :]}") # should be [[0 1 1 1 1 2 0 0 0 2]
                                                                                                                      #            [2 1 1 0 0 0 0 0 0 0]
                                                                                                                      #            [1 0 0 1 1 2 1 1 0 1]
                                                                                                                      #            [1 1 0 2 2 1 0 1 1 0]
                                                                                                                      #            [2 2 2 2 2 1 2 0 1 2]]
  test_predict = knn_model.predict(test_dataset)
  print(f"The predictions for test data are:\n {test_predict}") # should be [0 0 1 1 2 0 0 0 0 0 0 2 0 0 0 0 0 2 0 1 1 0 0 1 0 0 0 2 2 2 2 0 0 0 0 0 0
                                                                # 0 0 0 0 0 2 0 0 2 2 0 1 0 0 1 0 0 0 0 0 0 0 1 2 1 2 0 0 0 0 0 0 0 0 0 0 2
                                                                # 0 1 0 0 1 0 0 0 2 0 1 0 0 2 0 1 0 2 1 0 0 2 0 0 1 0]
  confusion_matrix = generate_confusion_matrix(test_predict, test_labels)
  print(f"The confusion matrix is:\n {confusion_matrix}") # should be [[48.  3.  1.]
                                                          #             [16. 11. 10.]
                                                          #             [ 4.  1.  6.]]
  accuracy = calculate_accuracy(test_predict, test_labels)
  print(f"The accuracy is: {accuracy}") # should be 0.65
  precision = calculate_precision(test_predict, test_labels)
  print(f"The macro average precision is: {precision}") # should be 0.5973856209150327
  recall = calculate_recall(test_predict, test_labels) 
  print(f"The macro average recall is: {recall}") # should be 0.5886095886095887
  macro_f1 = calculate_macro_f1(test_predict, test_labels)
  print(f"The macro f1 score is: {macro_f1}") # should be 0.5929651346720406
  MCC_score = calculate_MCC_score(test_predict, test_labels)
  print(f"The MCC score is: {MCC_score}") # should be 0.4182135132877802

  k_list = [5, 10, 15]
  p_list = [2, 4]
  d = 10
  dfoldcv = DFoldCV(train_dataset, train_labels, k_list, p_list, d, calculate_MCC_score)
  scores = dfoldcv.cross_validate()
  print(f"The scores for the first k value and the first p value: {scores[0, 0, :]}") # should be [0.35862701 0.44284459 0.32790457 0.39646162 0.21971336 0.3317104
                                                                                      #            0.27405523 0.3728344  0.391094   0.41420285]
  best_param = dfoldcv.validate_best_parameters()
  print(f"The best K value and p value are: {best_param}") # should be (10, 2)
  