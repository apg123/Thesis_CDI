## Nonused methods for parse in image search
"""
        if mode == 'paired_additions':
            VI=np.cov(vals, rowvar=False)
            sim_set_edit = sim_set.detach().clone().numpy()
            p_indices = [sim_set_edit[0]]
            sim_set_edit = np.delete(sim_set_edit, 0)
            p = [vals[0]]
            last_p =  [p[0]]
            vals = np.delete(vals, 0, axis=0)

            while len(p) < k:
                distances = cdist(vals, last_p, 'mahalanobis', VI=VI)
                min_distances = np.min(distances, axis=1)
                max_min = np.argmax(min_distances)
                #print(max_min)
                maximally_away = max_min
                p_indices.append(sim_set_edit[maximally_away])
                p.append(vals[maximally_away])
                sim_set_edit = np.delete(sim_set_edit, maximally_away)
                vals = np.delete(vals, maximally_away, axis=0)
                if len(p) < k:
                    distances = cdist(vals, p, 'mahalanobis', VI=VI)
                    mean_distances = np.mean(distances, axis=1)
                    max_mean = np.argmax(mean_distances)
                    #print(max_mean)
                    maximally_away = max_mean
                    p_indices.append(sim_set_edit[maximally_away])
                    last_p = [vals[maximally_away]]
                    p.append(vals[maximally_away])
                    sim_set_edit = np.delete(sim_set_edit, maximally_away)
                    vals = np.delete(vals, maximally_away, axis=0)

        if mode == 'convex_hull':
            ch = ConvexHull(vals)
            vert_indices = ch.vertices
            print(len(vert_indices))
            if len(vert_indices) > k:
                p_indices = np.random.choice(sim_set[vert_indices], k, replace = False)
            else:
                VI=np.cov(vals, rowvar=False)
                sim_set_edit = sim_set.detach().clone().numpy()
                p_indices = vert_indices.tolist()
                sim_set_edit = np.delete(sim_set_edit, p_indices)
                p = vals[p_indices]
                p = [x for x in p]
                vals = np.delete(vals, p_indices, axis=0)

                while len(p) < k:
                    distances = cdist(vals, p, 'mahalanobis', VI=VI)
                    mean_distances = np.mean(distances, axis=1)
                    #print(max_distances)
                    max_mean = np.argmax(mean_distances)
                    #print(max_max)
                    maximally_away = max_mean
                    p_indices.append(sim_set_edit[maximally_away])
                    p.append(vals[maximally_away])
                    sim_set_edit = np.delete(sim_set_edit, maximally_away)
                    vals = np.delete(vals, maximally_away, axis=0)
"""



## Old code for testing
"""
accs = []
pws = []
tws = []
tpws = []

d_accs = []
d_pws = []
d_tws = []
d_tpws = []

bad = 0
d_bad = 0

image_database = ImageDatabase(features, data, model, preprocess, device)
image_database.sensitive_attributes([("A picture of a man", "A picture of a woman"), ("A picture of a white person", "A picture of a black person"), ("A picture of a young person", "A picture of an old person")])


for cat in catagories:
    print(cat)
    res = image_database.search("This is a picture of a " + cat + ".", k = 25)
    n = res.search_term.value_counts()[cat]
    a = n / len(res)
    in_cat = res[res.search_term == cat]
    pw = len(in_cat[in_cat.image_gender == 'woman'])/ n

    if pw == 0 or pw == 1:
        bad += 1
    
    print(f"Rec: {a}, Precision function {precision('search_term', cat, res)}")
    print(f"Bias: {abs_bias_in_retrieval('search_term', [cat], res, 'image_gender', ['woman'], ['man'])}")
    #print(f"Skew: {skew_in_retrieval('search_term', [cat], res, 'image_gender', ['woman'], data[data['search_term'] == cat].iloc[0].bls_p_women)}")

    accs.append(a)
    pws.append(pw)
    tws.append(res.search_p_women.values[0])
    tpws.append(len(res[res.image_gender == 'woman']) / len(res))

    res = image_database.distinct_retrival("This is a picture of a " + cat + ".", k = 25)

    

    if cat in res.search_term.value_counts().keys():
        n = res.search_term.value_counts()[cat]
        in_cat = res[res.search_term == cat]
        pw = len(in_cat[in_cat.image_gender == 'woman'])/ n
        d_pws.append(pw)
        d_tws.append(res.search_p_women.values[0])
        if pw == 0 or pw == 1:
            d_bad += 1
    else:
        n = 0
        d_pws.append(0)
        d_tws.append(res.search_p_women.values[0])
    a = n / len(res)
    print(f"Rec: {a}, Precision function {precision('search_term', cat, res)}")
    print(f"Bias: {bias_in_retrieval('search_term', [cat], res, 'image_gender', ['woman'], ['man'])}")
    #print(f"Skew: {skew_in_retrieval('search_term', [cat], res, 'image_gender', ['woman'], data[data['search_term'] == cat].iloc[0].bls_p_women)}")
    d_tpws.append(len(res[res.image_gender == 'woman']) / len(res))
    d_accs.append(a)
    
    
    #print(f"Acc: {a:.2f}, PW: {pw:.2f}")
    #print('---')

plt.hist(accs)
plt.title("Accuracy for baseline search")
plt.xlabel(f"Accuracy: mean = {np.mean(accs):.2f}")
plt.show()

plt.hist(d_accs)
plt.title("Accuracy for diverse search")
plt.xlabel(f"Accuracy: mean = {np.mean(d_accs):.2f}")
plt.show()

plt.scatter(tws, pws)
plt.xlabel("True proportion of women in the images")
plt.ylabel("Retrieved proportion of women in the images")
plt.show()

plt.scatter(d_tws, d_pws)
plt.xlabel("True proportion of women in the images")
plt.ylabel("Retrieved proportion of women in the images")
plt.show()

print(f"Correlation Naive: {np.corrcoef(tws, pws)[0, 1]:.2f}; Correlation Distinct: {np.corrcoef(d_tws, d_pws)[0, 1]:.2f}")
print(f"Rate of Noninclusive results = {bad/len(pws):.2f}; Rate of Noninclusive results (Distinct) = {d_bad/len(d_pws):.2f}")

plt.scatter([x - y for x, y in zip(d_accs, accs)], [x - y for x, y in zip(d_pws, pws)])
plt.xlabel("Change in accuracy")
plt.ylabel("Change in proportion of women")
plt.show()

plt.scatter(tws, [x - y for x, y in zip(d_pws, pws)])
plt.xlabel("True proportion of women")
plt.ylabel("Change in proportion of women")
plt.show()

plt.scatter(tpws, d_tpws)
plt.xlabel
plt.show()"""

"""
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


image_database = ImageDatabase(features, data, model, preprocess, device)
image_database.sensitive_attributes([("A picture of a man", "A picture of a woman"), ("A picture of a white person", "A picture of a black person")]) #
image_database.define_pbm_classes(["unknown gender", "man", "woman"])

baseline_metrics = {'name': 'Baseline'}
random_from_similar_set_metrics = {'name': 'RandomSS'}
divimage_max_sum_metrics = {'name': 'DivImageMSum'}
divimage_max_min_metrics = {'name': 'DivImageMMin'}
random_from_similar_set_metrics_larger_tol = {'name': 'RandomSS'}
divimage_max_sum_metrics_larger_tol = {'name': 'DivImageMSum'}
divimage_max_min_metrics_larger_tol = {'name': 'DivImageMMin'}
pbm_metrics_gender = {'name': 'PBM_Gender'}
pbm_metrics_skintone = {'name': 'PBM_skintone'}
pbm_intersectional_metrics = {'name': 'PBM_Intersectional'}
pbm_intersectional_with_age_metrics = {'name': 'PBM_Intersectional_Age'}
debias_clip_metrics = {'name': 'DebiasClip'}


indistinguisable_values = [[cat] for cat in catagories]
totals_by_cat = {cat: len(data[data['search_term'] == cat]) for cat in catagories}
true_rates = [[data[data['search_term'] == cat].iloc[0].search_p_women for cat in catagories], [data[data['search_term'] == cat].iloc[0].search_p_dark for cat in catagories]]

print(true_rates)

#run_analysis(lambda x: image_database.search(x, 25), 25, None, baseline_metrics, catagories, 'search_term', indistinguisable_values, ['gender'], [['Female']], [['Male']], true_rates)
#df = parse_analysis([baseline_metrics], ['gender'])
#print(df)

run_analysis(lambda x: image_database.search(x, 25), 25, None, baseline_metrics, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
run_analysis(lambda x: image_database.distinct_retrival(x, 25, tol=.02, method='max_sum'), 25, .02, divimage_max_sum_metrics, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
run_analysis(lambda x: image_database.distinct_retrival(x, 25, tol=.02, method='max_min'), 25, .02, divimage_max_min_metrics, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
run_analysis(lambda x: image_database.distinct_retrival(x, 25, tol=.02, method='random'), 25, .02, random_from_similar_set_metrics, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
run_analysis(lambda x: image_database.distinct_retrival(x, 25, tol=.04, method='max_sum'), 25, .04, divimage_max_sum_metrics_larger_tol, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
run_analysis(lambda x: image_database.distinct_retrival(x, 25, tol=.04, method='max_min'), 25, .04, divimage_max_min_metrics_larger_tol, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
run_analysis(lambda x: image_database.distinct_retrival(x, 25, tol=.04, method='random'), 25, .04, random_from_similar_set_metrics_larger_tol, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
## Add with age
## Add with age


image_database.define_pbm_classes(["unknown gender", "man", "woman"])
run_analysis(lambda x: image_database.pbm(x, 25, eps=0), 25, None, pbm_metrics_gender, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
image_database.define_pbm_classes(["unknown skin-tone", "light-skinned person", "dark-skinned person"])
run_analysis(lambda x: image_database.pbm(x, 25, eps=0), 25, None, pbm_metrics_skintone, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
image_database.define_pbm_classes(["unknown gender and skin-tone", "light-skinned man", "light-skinned woman", "dark-skinned man", "dark-skinned woman"])
run_analysis(lambda x: image_database.pbm(x, 25, eps=0), 25, None, pbm_intersectional_metrics, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)
image_database.define_pbm_classes(["unknown gender and skin-tone", "light-skinned old man", "light-skinned old woman", "dark-skinned old man", "dark-skinned old woman", "light-skinned young man", "light-skinned young woman", "dark-skinned young man", "dark-skinned young woman"])
run_analysis(lambda x: image_database.pbm(x, 25, eps=0), 25, None, pbm_intersectional_with_age_metrics, catagories, 'search_term', indistinguisable_values, ['gender', 'skintone'], [['Female'], ['dark']], [['Male'], ['light']], true_rates, totals_by_cat)

#run_analysis(lambda x: debias_database.search(x, 25), 25, None, debias_clip_metrics, catagories, 'search_term', indistinguisable_values, ['image_gender'], [['woman']], [['man']], true_rates)

df = parse_analysis([baseline_metrics, divimage_max_sum_metrics, divimage_max_min_metrics, random_from_similar_set_metrics, divimage_max_sum_metrics_larger_tol, divimage_max_min_metrics_larger_tol, random_from_similar_set_metrics_larger_tol, pbm_metrics_gender, pbm_metrics_skintone, pbm_intersectional_metrics, pbm_intersectional_with_age_metrics], ['gender', 'skintone']) #debias_clip_metrics
plt.hist(baseline_metrics['precision'] - divimage_metrics['precision'])
plt.show()

plt.hist(baseline_metrics['bias'] - divimage_metrics['bias'])
plt.show()"""

#print(f"Baseline Average Bias: {np.mean(baseline_metrics['bias'])}, DivImage Average Bias: {np.mean(divimage_metrics['bias'])}")

#print(divimage_metrics['bias'])"""