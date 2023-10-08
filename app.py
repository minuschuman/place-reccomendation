from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator

app = Flask(__name__)

with open('dataset/user_similarity.pkl', 'rb') as user_sim_file:
    user_similarity = pickle.load(user_sim_file)

with open('dataset/item_similarity.pkl', 'rb') as item_sim_file:
    item_similarity = pickle.load(item_sim_file)

with open('dataset/piv_norm.pkl', 'rb') as piv_norm_file:
    piv_norm = pickle.load(piv_norm_file)


@app.route('/recommend/top_locations/<location_name>', methods=['GET'])
def recommend_top_locations(location_name):
    count = 1
    similar_locations = []
    for item in item_similarity.sort_values(by=location_name, ascending=False).index[1:11]:
        similar_locations.append(item)
        count += 1
    return jsonify({'top_locations': similar_locations})

@app.route('/recommend/top_users/<user_id>', methods=['GET'])
def recommend_top_users(user_id):
    if user_id not in piv_norm.columns:
        return jsonify({'message': 'No data available on user {}'.format(user_id)})
    similar_users = []
    sim_values = user_similarity.sort_values(by=user_id, ascending=False).loc[:, user_id].tolist()[1:11]
    sim_users = user_similarity.sort_values(by=user_id, ascending=False).index[1:11]
    zipped = zip(sim_users, sim_values)
    for user, sim in zipped:
        similar_users.append({'user_id': user, 'similarity_value': sim})
    return jsonify({'top_users': similar_users})

@app.route('/recommend/similar_user_recs/<user_id>', methods=['GET'])
def recommend_similar_user_recs(user_id):
    if user_id not in piv_norm.columns:
        return jsonify({'message': 'No data available on user {}'.format(user_id)})
    similar_users = user_similarity.sort_values(by=user_id, ascending=False).index[1:11]
    best = []
    most_common = {}
    for i in similar_users:
        max_score = piv_norm.loc[:, i].max()
        best.append(piv_norm[piv_norm.loc[:, i] == max_score].index.tolist())
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1
    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
    top_recommendations = sorted_list[:5]
    return jsonify({'similar_user_recs': top_recommendations})

if __name__ == '__main__':
    app.run(debug=True)
