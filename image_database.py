from helpers import *
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import random

class ImageDatabase():
    def __init__(self, features, data, model, preprocess, device="cuda"):
        if device == "cuda": self.arraytype = torch.float16 
        else: self.arraytype = torch.float32
        self.features = t_normalize(torch.tensor(features).to(device)).to(self.arraytype)
        self.data = data
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.clipclip_orderings = {}

    def _process_query(self, query):
        token = clip.tokenize(query).to(self.device)
        #print(token.device)
        with torch.no_grad():
            query_features = t_normalize(self.model.encode_text(token))
        return query_features
        
    def search(self, query, k=10, **kwargs):
        query_features = self._process_query(query)
        similarities = (self.features @ query_features.T).flatten()
        best = similarities.argsort(descending=True).cpu().flatten()
        return self.data.iloc[best[:k]]
    
    def sensitive_attributes(self, paired_attributes):
        l_attr = list(sum(paired_attributes, ()))
        tokens = clip.tokenize(l_attr).to(self.device)
        with torch.no_grad():
            attributes_features = t_normalize(self.model.encode_text(tokens))
        self.sensitive_ideals = attributes_features.reshape(len(paired_attributes), 2, -1)

        return self
        
    def _calc_sim_set(self, best, similarities, k, max_sim_set, tol):
        sim = similarities[best]
        sim_top_k_avg = sim[0:k].mean()
        for i in range(len(sim)):
            if sim[i] < sim_top_k_avg - tol:
                break
        k = min(max(k, i), max_sim_set)
        return best[0:k]


    def _get_sim_to_ideal(self, entries):
        sim_set_features = self.features[entries]
        sim = torch.matmul(self.sensitive_ideals, sim_set_features.T)
        proba = (100 * sim).permute(2, 0, 1).softmax(dim=-1)[:, :, 0].to('cpu').numpy()
        return proba

    def _retrieve_distinct(self, sim_set, similarities, k, mode='max_sum'):
        proba_concepts = self._get_sim_to_ideal(sim_set)
        vals = proba_concepts
        pca = PCA()
        
        if mode == 'max_sum': 
            VI=np.cov(vals, rowvar=False)
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            vals = np.delete(vals, 0, axis=0)

            distances = cdist(vals, p, 'mahalanobis', VI=VI)

            while len(p) < k:
                mean_distances = np.mean(distances, axis=1)
                max_sum = np.argmax(mean_distances)
                maximally_away = max_sum
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                distances = np.hstack((distances, cdist(vals, [vals[maximally_away]], 'mahalanobis', VI=VI)))
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                distances = np.delete(distances, maximally_away, axis=0)
        
        if mode == 'euc_max_sum': 
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            vals = np.delete(vals, 0, axis=0)

            distances = cdist(vals, p, 'euclidean')

            while len(p) < k:
                mean_distances = np.mean(distances, axis=1)
                max_sum = np.argmax(mean_distances)
                maximally_away = max_sum
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                distances = np.hstack((distances, cdist(vals, [vals[maximally_away]], 'euclidean')))
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                distances = np.delete(distances, maximally_away, axis=0)

        if mode == 'max_min':
            VI=np.cov(vals, rowvar=False)
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            vals = np.delete(vals, 0, axis=0)

            distances = cdist(vals, p, 'mahalanobis', VI=VI)

            while len(p) < k:
                min_distances = np.min(distances, axis=1)
                max_min = np.argmax(min_distances)
                maximally_away = max_min
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                distances = np.hstack((distances, cdist(vals, [vals[maximally_away]], 'mahalanobis', VI=VI)))
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                distances = np.delete(distances, maximally_away, axis=0)
        
        if mode == 'euc_max_min':
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            vals = np.delete(vals, 0, axis=0)

            distances = cdist(vals, p, 'euclidean')

            while len(p) < k:
                min_distances = np.min(distances, axis=1)
                max_min = np.argmax(min_distances)
                maximally_away = max_min
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                distances = np.hstack((distances, cdist(vals, [vals[maximally_away]], 'euclidean')))
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                distances = np.delete(distances, maximally_away, axis=0)

        if mode == 'random':
            p_indices = np.random.choice(sim_set, k, replace = False)
            
            
        if mode == "feature_distances":
            vals = torch.index_select(self.features, 0, sim_set.to(self.device)).cpu().numpy()
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            vals = np.delete(vals, 0, axis=0)

            distances = cdist(vals, p, 'cosine')

            while len(p) < k:
                mean_distances = np.mean(distances, axis=1)
                max_sum = np.argmax(mean_distances)
                maximally_away = max_sum
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                distances = np.hstack((distances, cdist(vals, [vals[maximally_away]], 'cosine')))
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                distances = np.delete(distances, maximally_away, axis=0)

        if mode == 'true_labels':
            vals = self._get_true_coordinates(sim_set)
            VI=np.cov(vals, rowvar=False)
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            vals = np.delete(vals, 0, axis=0)

            distances = cdist(vals, p, 'mahalanobis', VI=VI)

            while len(p) < k:
                mean_distances = np.mean(distances, axis=1)
                max_sum = np.argmax(mean_distances)
                maximally_away = max_sum
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                distances = np.hstack((distances, cdist(vals, [vals[maximally_away]], 'mahalanobis', VI=VI)))
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                distances = np.delete(distances, maximally_away, axis=0)

        return p_indices
    
    def define_coordinate_mapping(self, columns, positive_labels, negative_labels):
        self.true_coordinates = np.zeros((len(self.data), len(columns)))
        self.coord_columns = columns
        for i, column in enumerate(columns):
            map_to_hypercube = lambda x: 1 if x in positive_labels[i] else (0 if x in negative_labels[i] else .5)
            self.true_coordinates[:, i] = self.data[column].apply(map_to_hypercube)
    
    def _get_true_coordinates(self, sim_set):
        return self.true_coordinates[sim_set] 

    
    def distinct_retrival(self, query, k=10, max_sim_set=1000, tol=.06, method='max_sum', **kwargs) :
        query_features = self._process_query(query)

        similarities = (self.features @ query_features.T).flatten()
        best = similarities.argsort(descending=True).cpu().flatten()

        sim_set = self._calc_sim_set(best, similarities, k, max_sim_set, tol)
        distinct_sort = self._retrieve_distinct(sim_set, similarities, k, mode=method)


        return self.data.iloc[distinct_sort]

    def define_pbm_classes(self, classes):
        self.pbm_classes=classes
        prompts = [f"A picture of a {c}." for c in classes]
        if classes[0] == "empty":
            prompts[0] == ""
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            attributes_features = t_normalize(self.model.encode_text(tokens))
        self.pbm_ideals = attributes_features
        self.pbm_label = np.argmax((100 * torch.matmul(self.features, self.pbm_ideals.T)).softmax(dim=-1).to('cpu').numpy(), axis=-1)

        return self

    def pbm(self, query, k=10, eps=0, **kwargs):
        ## As defined in the paper "Mitigating Test-Time Bias for Fair Image Retrieval" (Kong et. al. 2023)
        query_features = self._process_query(query)
        similarities = (self.features @ query_features.T).flatten()
        best = similarities.argsort(descending=True).cpu().numpy().flatten()
        np_sim = similarities.cpu().numpy()

        p_indices = []

        neutrals = [x for x in best if self.pbm_label[x] == 0]
        classes = [[x for x in best if self.pbm_label[x]== i] for i in range(1, len(self.pbm_classes))]

    
        while len(p_indices) < k:
            if random.random() < eps:
                try:
                    neutral_sim = np_sim[neutrals[0]]
                except:
                    neutral_sim = -1
                
                max_class, idx = 0, 0
                for i, c in enumerate(classes):
                    try:
                        class_sim = np_sim[c[0]]
                    except:
                        class_sim = -1
                    if class_sim > max_class:
                        max_class = class_sim
                        idx = i
                if max_class > neutral_sim:
                    p_indices.append(classes[idx][0])
                    classes[idx].pop(0)
                else:
                    p_indices.append(neutrals[0])
                    neutrals.pop(0)
                        
            else:
                best_neutral = neutrals[0]
                best_for_classes = [fon(c) for c in classes]
                best_for_classes_vals = [c for c in best_for_classes if c is not None]

                similarities_for_classes = [np_sim[x] for x in best_for_classes_vals]
                avg_sim = np.mean(similarities_for_classes)
                neutral_sim = similarities[best_neutral]

                if avg_sim > neutral_sim:
                    if len(p_indices) + len(best_for_classes_vals) > k:
                        best_for_classes_vals = random.choices(best_for_classes_vals, k=k-len(p_indices))
                    p_indices += best_for_classes_vals

                    for i, x in enumerate(best_for_classes):
                        if x is not None:
                            classes[i].pop(0)
                else:
                    p_indices.append(best_neutral)
                    neutrals.pop(0)
        
        return self.data.iloc[p_indices]

    def add_clipclip_ordering(self, name, ordering):
        self.clipclip_orderings[name] = ordering.copy()
        return self
    
    def clip_clip(self, query, ordering, n_to_clip, k=10, **kwargs):
        # As defined in the paper "Are Gender-Neutral Queries Really Gender-Neutral? Mitigating Gender Bias in Image Search" (Wang et. al. 2021)
        query_features = self._process_query(query)
        clip_ordering = self.clipclip_orderings[ordering]
        clip_features = torch.index_select(self.features, 1, torch.tensor(clip_ordering[n_to_clip:]).to(self.device))
        clip_query = torch.index_select(query_features, 1, torch.tensor(clip_ordering[n_to_clip:]).to(self.device))

        similarities = (clip_features @ clip_query.T).flatten()
        best = similarities.argsort(descending=True).cpu().flatten()
        return self.data.iloc[best[:k]]

def run_analysis(method_call, k, tol, result_dict, catagories, catagorical_column, label_indistinguishable_values_list, protected_columns, protected_positive_values, protected_negative_values, true_rates, totals_by_cat, prefix="This is a picture of a"):
    result_dict['k'] = k
    result_dict['tol'] = tol
    result_dict['precision'] = np.zeros(len(catagories))
    result_dict['precision_up_to_indistinguishablity'] = np.zeros(len(catagories))
    result_dict['recall'] = np.zeros(len(catagories))
    result_dict['bias'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['abs_bias'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['skew'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['bias_for_accurate'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['abs_bias_for_accurate'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['skews_for_accurate'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['max_skew'] = np.zeros(len(catagories))
    result_dict['min_skew'] = np.zeros(len(catagories))
    result_dict['worst_multiclass_error'] = np.zeros(len(catagories))



    for i, cat in enumerate(catagories):
        res = method_call(f"{prefix} {cat}")
        result_dict['precision'][i] = precision(label_column=catagorical_column, positive_label_value=cat, data=res)
        result_dict['precision_up_to_indistinguishablity'][i] = precision_up_to_indistinguishablity(label_column=catagorical_column, indistinguishable_labels=label_indistinguishable_values_list[i], data=res)
        result_dict['recall'][i] = recall(label_column=catagorical_column, positive_label_value=cat, data=res, total_positive=totals_by_cat[cat])
        biases = np.zeros(len(protected_columns))
        skews = np.zeros(len(protected_columns))
        abs_biases = np.zeros(len(protected_columns))
        biases_for_accurate = np.zeros(len(protected_columns))
        abs_biases_for_accurate = np.zeros(len(protected_columns))
        skews_for_accurate = np.zeros(len(protected_columns))

        for j, (protected_col, protected_pos_vals, protected_neg_vals) in enumerate(zip(protected_columns, protected_positive_values, protected_negative_values)):
            abs_bias, abs_bias_in_correct_retrievals = abs_bias_in_retrieval(label_column=catagorical_column, positive_label_values=[cat], data=res, protected_column=protected_col, protected_positive_values=protected_pos_vals, protected_negative_values=protected_neg_vals)
            bias, bias_in_correct_retrievals = bias_in_retrieval(label_column=catagorical_column, positive_label_values=[cat], data=res, protected_column=protected_col, protected_positive_values=protected_pos_vals, protected_negative_values=protected_neg_vals)
            biases[j] = bias
            biases_for_accurate[j] = bias_in_correct_retrievals
            abs_biases[j] = abs_bias
            abs_biases_for_accurate[j] = abs_bias_in_correct_retrievals
            skew, skew_for_accurate = skew_in_retrieval(label_column=catagorical_column, positive_label_values=[cat], data=res, protected_column=protected_col, protected_positive_values=protected_pos_vals, true_rate=true_rates[j][i])
            skews[j] = skew
            skews_for_accurate[j] = skew_for_accurate
        
        result_dict['bias'][i] = biases
        result_dict['abs_bias'][i] = abs_biases
        result_dict['skew'][i] = skews
        result_dict['bias_for_accurate'][i] = biases_for_accurate
        result_dict['abs_bias_for_accurate'][i] = abs_biases_for_accurate
        result_dict['skews_for_accurate'][i] = skews_for_accurate
        result_dict['max_skew'][i] = max(skews)
        result_dict['min_skew'][i] = min(skews)
        result_dict['worst_multiclass_error'][i] = multiclass_bias_in_retrieval(label_column=catagorical_column, positive_label_values=[cat], data=res, protected_columns=protected_columns, protected_positive_values=protected_positive_values, protected_negative_values=protected_negative_values)

    return result_dict

def parse_analysis(result_dicts, protected_column_names):
    data = []
    for result in result_dicts:
        parsed_result = {}
        if result['tol'] is not None:
            if result['name'][0:3] == "PBM":
                parsed_result['method'] = result['name'] + "(eps " + str(result['tol']) + ")"
            parsed_result['method'] = result['name'] + " (tol: " + str(result['tol']) + ")"
        else:
            parsed_result['method'] = result['name']
        parsed_result['name'] = result['name']
        parsed_result['tol'] = result['tol']
        parsed_result['k'] = result['k']
        parsed_result['Avg_Precision'] = np.mean(result['precision'])
        parsed_result['Avg_Recall'] = np.mean(result['recall'])
        parsed_result['Avg_PutI'] = np.mean(result['precision_up_to_indistinguishablity'])

        for i, protected_col in enumerate(protected_column_names):
            parsed_result[f'Avg_AbsBias_{protected_col}'] = np.mean(result['abs_bias'][:,i])
            parsed_result[f'Avg_Bias_{protected_col}'] = np.mean(result['bias'][:,i])
            parsed_result[f'Avg_Skew_{protected_col}'] = np.mean(result['skew'][:,i])
            parsed_result[f'Avg_Abs_Skew_{protected_col}'] = np.mean(np.abs(result['skew'][:,i]))
            parsed_result[f'Avg_AbsBias_for_Accurate_{protected_col}'] = np.mean(result['abs_bias_for_accurate'][:,i])

        parsed_result['Avg_Max_MC_Bias'] =np.mean(result['worst_multiclass_error'])

        if len(protected_column_names) > 1:
            parsed_result['Max_AbsBias'] = np.max([np.mean(result['abs_bias'][:,i]) for i in range(len(protected_column_names))])

        data.append(parsed_result.copy())

    df = pd.DataFrame(data)
    return df


def run_analysis_celeba(method_call, k, tol, result_dict, catagories, catagory_columns, label_indistinguishable_values_list, protected_columns, protected_positive_values, protected_negative_values, totals_by_cat, prefix="This is a picture of a"):
    result_dict['k'] = k
    result_dict['tol'] = tol
    result_dict['precision'] = np.zeros(len(catagories))
    result_dict['precision_up_to_indistinguishablity'] = np.zeros(len(catagories))
    result_dict['recall'] = np.zeros(len(catagories))
    result_dict['bias'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['abs_bias'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['bias_for_accurate'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['abs_bias_for_accurate'] = np.zeros((len(catagories), len(protected_columns)))
    result_dict['worst_multiclass_error'] = np.zeros(len(catagories))



    for i, cat in enumerate(catagories):
        res = method_call(f"{prefix} {cat}")
        result_dict['precision'][i] = precision(label_column=catagory_columns[i], positive_label_value="1", data=res)
        result_dict['precision_up_to_indistinguishablity'][i] = precision_up_to_indistinguishablity(label_column=catagory_columns[i], indistinguishable_labels=label_indistinguishable_values_list[i], data=res)
        result_dict['recall'][i] = recall(label_column=catagory_columns[i], positive_label_value="1", data=res, total_positive=totals_by_cat[cat])
        biases = np.zeros(len(protected_columns))
        abs_biases = np.zeros(len(protected_columns))
        biases_for_accurate = np.zeros(len(protected_columns))
        abs_biases_for_accurate = np.zeros(len(protected_columns))

        for j, (protected_col, protected_pos_vals, protected_neg_vals) in enumerate(zip(protected_columns, protected_positive_values, protected_negative_values)):
            abs_bias, abs_bias_in_correct_retrievals = abs_bias_in_retrieval(label_column=catagory_columns[i], positive_label_values=['1'], data=res, protected_column=protected_col, protected_positive_values=protected_pos_vals, protected_negative_values=protected_neg_vals)
            bias, bias_in_correct_retrievals = bias_in_retrieval(label_column=catagory_columns[i], positive_label_values=['1'], data=res, protected_column=protected_col, protected_positive_values=protected_pos_vals, protected_negative_values=protected_neg_vals)
            biases[j] = bias
            biases_for_accurate[j] = bias_in_correct_retrievals
            abs_biases[j] = abs_bias
            abs_biases_for_accurate[j] = abs_bias_in_correct_retrievals
        
        result_dict['bias'][i] = biases
        result_dict['abs_bias'][i] = abs_biases
        result_dict['bias_for_accurate'][i] = biases_for_accurate
        result_dict['abs_bias_for_accurate'][i] = abs_biases_for_accurate
        result_dict['worst_multiclass_error'][i] = multiclass_bias_in_retrieval(label_column=catagory_columns[i], positive_label_values=['1'], data=res, protected_columns=protected_columns, protected_positive_values=protected_positive_values, protected_negative_values=protected_negative_values)

    return result_dict

def parse_analysis_celeba(result_dicts, protected_column_names):
    data = []
    for result in result_dicts:
        parsed_result = {}
        if result['tol'] is not None:
            if result['name'][0:3] == "PBM":
                parsed_result['method'] = result['name'] + "(eps " + str(result['tol']) + ")"
            parsed_result['method'] = result['name'] + " (tol: " + str(result['tol']) + ")"
        else:
            parsed_result['method'] = result['name']
        parsed_result['name'] = result['name']
        parsed_result['tol'] = result['tol']
        parsed_result['k'] = result['k']
        parsed_result['Avg_Precision'] = np.mean(result['precision'])
        parsed_result['Avg_Recall'] = np.mean(result['recall'])
        parsed_result['Avg_PutI'] = np.mean(result['precision_up_to_indistinguishablity'])

        for i, protected_col in enumerate(protected_column_names):
            parsed_result[f'Avg_AbsBias_{protected_col}'] = np.mean(result['abs_bias'][:,i])
            parsed_result[f'Avg_Bias_{protected_col}'] = np.mean(result['bias'][:,i])
            parsed_result[f'Avg_AbsBias_for_Accurate_{protected_col}'] = np.mean(result['abs_bias_for_accurate'][:,i])

        parsed_result['Avg_Max_MC_Bias'] =np.mean(result['worst_multiclass_error'])

        if len(protected_column_names) > 1:
            parsed_result['Max_AbsBias'] = np.max([np.mean(result['abs_bias'][:,i]) for i in range(len(protected_column_names))])

        data.append(parsed_result.copy())

    df = pd.DataFrame(data)
    return df