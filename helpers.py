import random
import itertools
import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
import clip
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


class ZeroShotClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classes, model_preprocess = None, temp = 100, prefix = ""):
        self.classes_ = list(range(len(classes)))
        self.temp = temp
        self.prefix = prefix
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_preprocess is None:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        else:
            self.model, self.preprocess = model_preprocess

        class_tokens = clip.tokenize([self.prefix + a for a in classes]).to(self.device)

        with torch.no_grad():
            self.vect_classes = t_normalize(self.model.encode_text(class_tokens))
        
        self.is_fitted_ = True

    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def _t_normalize(self, X):
        return X / X.norm(dim=-1, keepdim=True)
    
    def _embed_text(self, X):
        #TODO Batching
        token = clip.tokenize(X).to(self.device)
        with torch.no_grad():
            embedding = self._t_normalize(self.model.encode_text(token))
        return embedding
        
    def _embed_image(self, X):
        #TODO Batching
        X_tensor = torch.cat([self.preprocess(x).unsqueeze(0).to('cuda') for x in X])
        with torch.no_grad():
            embedding = self._t_normalize(self.model.encode_image(X_tensor))
        return embedding

    def predict_proba(self, X):
        try:
            ex = X[0]
        except:
            raise ValueError("Must provide an iterable to predict.")
        if isinstance(ex, str):
            embedding_function = self._embed_text
        elif isinstance(ex, Image.Image):
            embedding_function = self._embed_image
        else:
            raise ValueError("Must provide either text or images.")
        
        embedding = embedding_function(X)

        proba = (self.temp * embedding @ self.vect_classes.T).softmax(dim=-1)

        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X).cpu().numpy()
        classes = np.argmax(proba, axis=1)
        return classes
    
class DifferenceSubspace():

    def __init__(self, X = None, pca_method = None):
        self.X_ = X
        self.pca_method = None

    def gen_from_pairs(self, pairs):
        if torch.is_tensor(pairs):
            pairs = pairs.cpu().numpy()
        self.X_ = pairs[:, 1] - pairs[:, 0]
        self.build_pca()
        return self
    
    def build_pca(self):
        if self.pca_method is None:
            self.pca_ = PCA()
        else:
            self.pca_ = self.pca_method

        if self.X_ is not None:
            self.pca_.fit(self.X_)
        else:
            raise ValueError("No X_ has been provided")
    
    def mahalanobis_distance(self, X):
        if not hasattr(self, 'pca_'):
            try:
                self.build_pca()
            except:
                raise ValueError("No PCA has been fit!")
        
        return np.linalg.norm(self.pca_.transform(X), axis=1)
        

def t_normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

def random_combinations(a, b, k):
    return random.sample(list(itertools.product(a, b)), k)

def influence_information(combs):
    return [(a, b, "This is a picture of " + b + " " + a) for a, b in combs]


def covering_influence_space(pca, X, alpha = .05):
    proj_influence = torch.tensor(pca.inverse_transform(pca.transform(X)))
    sim = np.sum(normalize(proj_influence) * t_normalize(X).numpy(), axis = 1)
    percent =  (sim > 1 - alpha).sum() / len(X)
    return percent > (1 - alpha)

def covering_influece_components_search(COMP, X, alpha = .05):
    n_features = len(X[0])
    n_samples = len(X)
    upper, lower = min(n_features, n_samples), 0
    found = False

    while not found:
        guess = int((upper + lower) / 2)
        pca = COMP(n_components = guess)
        pca.fit(X)
        if covering_influence_space(pca, X, alpha):
            upper = guess
        else:
            lower = guess + 1
        if lower == upper:
            found = True
    return upper

def process_images(model, preprocess, image_locations, device = 'cuda', batch_size=200):
    all_features = []
    for i in tqdm(range(0, len(image_locations), batch_size)):
        try:
            images = torch.cat([preprocess(Image.open(img)).unsqueeze(0)for img in image_locations[i:i+batch_size]]).to(device)
        except:
            print("Error with images, preprossing each image individually")
            print("Error with image", img)
            good_images = []
            for img in image_locations[i:i+batch_size]:
                try:
                    good_images.append(preprocess(Image.open(img)).unsqueeze(0))
                except:
                    print("Error with image", img)
                    continue
            images = torch.cat(good_images).to(device)
        with torch.no_grad():
            features = model.encode_image(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)

    
def precision(label_column, positive_label_value, data):
    return (data[label_column] == positive_label_value).sum() / len(data)

def precision_up_to_indistinguishablity(label_column, indistinguishable_labels, data):
    return (data[label_column].isin(indistinguishable_labels)).sum() / len(data)

def recall(label_column, positive_label_value, data, total_positive):
    return (data[label_column] == positive_label_value).sum() / total_positive

def abs_bias_in_retrieval(label_column, positive_label_values, data, protected_column, protected_positive_values, protected_negative_values):
    bias = abs(precision_up_to_indistinguishablity(protected_column, protected_positive_values, data) - precision_up_to_indistinguishablity(protected_column, protected_negative_values, data))
    #print(bias)
    data_with_correct_class = data[data[label_column].isin(positive_label_values)]
    if len(data_with_correct_class) == 0:
        return bias, 0
    bias_in_correct_class = abs(precision_up_to_indistinguishablity(protected_column, protected_positive_values, data_with_correct_class) - precision_up_to_indistinguishablity(protected_column, protected_negative_values, data_with_correct_class))
    return bias, bias_in_correct_class

def bias_in_retrieval(label_column, positive_label_values, data, protected_column, protected_positive_values, protected_negative_values):
    bias = precision_up_to_indistinguishablity(protected_column, protected_positive_values, data) - precision_up_to_indistinguishablity(protected_column, protected_negative_values, data)
    #print(bias)
    data_with_correct_class = data[data[label_column].isin(positive_label_values)]
    if len(data_with_correct_class) == 0:
        return bias, 0
    bias_in_correct_class = precision_up_to_indistinguishablity(protected_column, protected_positive_values, data_with_correct_class) - precision_up_to_indistinguishablity(protected_column, protected_negative_values, data_with_correct_class)
    return bias, bias_in_correct_class

def skew_in_retrieval(label_column, positive_label_values, data, protected_column, protected_positive_values, true_rate, eps =.001):
    a = precision_up_to_indistinguishablity(protected_column, protected_positive_values, data) / (true_rate + eps)
    if a == 0:
        if true_rate == 0:
            return 0, 0
        else:
            a = (1 / len(data)) / true_rate
    skew =  math.log(a)

    data_with_correct_class = data[data[label_column].isin(positive_label_values)]

    if len(data_with_correct_class) == 0:
        return skew, np.nan
    
    b = (precision_up_to_indistinguishablity(protected_column, protected_positive_values, data_with_correct_class) + eps)  / (true_rate + eps)

    if b == 0:
        if true_rate == 0:
            return 0, 0
        else:
            b = (1 / len(data_with_correct_class) + eps) / (true_rate)
    skew_in_correct_class =  math.log(b)
    return skew, skew_in_correct_class

def max_skew_in_retrieval(label_column, positive_label_values, data, protected_columns, protected_positive_values, true_rates):
    max = 1
    max_in_correct_class = 1
    max_column = None
    max_column_correct_class = None

    for column, values, rate in zip(protected_columns, protected_positive_values, true_rates):
        skew, skew_in_correct_class = skew_in_retrieval(label_column, positive_label_values, data, column, values, rate)
        if skew > max:
            max = skew
            max_column = column
        if skew_in_correct_class > max_in_correct_class:
            max_in_correct_class = skew_in_correct_class
            max_column_correct_class = column

    return max, max_column, max_in_correct_class, max_column_correct_class

def min_skew_in_retrieval(label_column, positive_label_values, data, protected_columns, protected_positive_values, true_rates):
    max = 1
    max_in_correct_class = 1
    max_column = None
    max_column_correct_class = None

    for column, values, rate in zip(protected_columns, protected_positive_values, true_rates):
        skew, skew_in_correct_class = skew_in_retrieval(label_column, positive_label_values, data, column, values, rate)
        if skew < max:
            max = skew
            max_column = column
        if skew_in_correct_class > max_in_correct_class:
            max_in_correct_class = skew_in_correct_class
            max_column_correct_class = column

    return max, max_column, max_in_correct_class, max_column_correct_class

def multiclass_bias_in_retrieval(label_column, positive_label_values, data, protected_columns, protected_positive_values, protected_negative_values):
    df = data.copy()
    df['protected'] = df[protected_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
    
    res = []

    for code in itertools.product(range(2), repeat = len(protected_columns)):
        labels = []
        for i, c in enumerate(code):
            if c == 0:
                labels.append(protected_positive_values[i])
            else:
                labels.append(protected_negative_values[i])
        res.append(precision_up_to_indistinguishablity('protected', [''.join(x) for x in itertools.product(*labels, repeat = 1)], df))
    
    return abs(np.ptp(res))


def fon(l):
    try:
        return l[0]
    except:
        return None
    
def unpack_to_i(l, i):
    if type(l) != list:
        return l[i]
    else:
        return [x[i] for x in l]
        
def intersections_for_pbm(pairs, unknown, postfix):
    all = [unknown]
    for prod in itertools.product(*pairs):
        all.append("a " + " ".join(prod) + postfix)
    return all

def plot_across_tol(df, k, method_names, axis1, axis2, xlabel=None, ylabel=None, title=None, reverse_x = False):
    relevant_k = df[df['k'] == k]

    for method in method_names:
        data_for_method = relevant_k[relevant_k['name'] == method]
        d1 = data_for_method[axis1]
        d2 = data_for_method[axis2]

        if method in ["Baseline", "DebiasClip"]:
            plt.scatter(d1, d2, label=method)
        else:
            plt.plot(d1, d2, label=method)
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(axis1)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(axis2)
    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Graph of {axis2} over {axis1}")
    if reverse_x:
        plt.gca().invert_xaxis()
    plt.grid()
    plt.show()
    
    