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