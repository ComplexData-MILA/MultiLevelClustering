'''
Author: Pratheeksha Nair

This file contains code for detecting organized groups from twitter data pertaining to 'OnlyFans'
Provide a path to pickle files containing the filtered community information as input to this code
To run:
	python multilevel_clustering.py --communities_path /path/to/communities/ 
'''
import pickle as pkl
import numpy as np
import pandas as pd
from utils import *
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import TruncatedSVD
import os
from scipy.sparse import lil_matrix
import hdbscan

pd.options.mode.chained_assignment = None  # default='warn'

cluster_count = {'2017': 30,
				 '2018': 40,
				 '2019': 80,
				 '2020': 250,
				 '2021': 230}

def common_handles(full_name_feats, person_to_id_map):
	common_persons = {}
	for year in full_name_feats.keys():
		col_sum = np.sum(full_name_feats[year].todense(), axis=0)
		# print(full_name_feats[year].shape, col_sum.shape)
		print(np.where(col_sum>1))
		common_persons[year] = np.where(col_sum>1)[1]

	names_in_all_years = set(common_persons['2017']) & set(common_persons['2018']) & \
				set(common_persons['2019']) & set(common_persons['2020']) & set(common_persons['2021'])
	print(names_in_all_years)
	clusters_with_common_handles = {}
	for year in full_name_feats.keys():
		rows_containing_common_handles = full_name_feats[year].todense()[:,np.array(list(names_in_all_years))]
		# print(rows_containing_common_handles)
		common_locs = np.where(rows_containing_common_handles==1)[0]
		clusters_with_common_handles[year] = common_locs

	id_to_person_map = {ind: per for per, ind in person_to_id_map.items()}
	person_names = [id_to_person_map[nm] for nm in names_in_all_years]
	print(clusters_with_common_handles)
	return person_names, clusters_with_common_handles

def account_mentions():
	if os.path.exists("cluster_acc_mentions.pkl"):
		feat_mats = pkl.load(open("cluster_acc_mentions.pkl",'rb'))
		person_to_id_map = pkl.load(open("person_to_id_map.pkl",'rb'))
		return feat_mats, person_to_id_map

	curr = 0
	full_vocab_ind = {}
	all_mentioned_persons = set()
	mentions = {}
	for year in tqdm(['2017','2018','2019','2020','2021']):
		persons_mentioned = pkl.load(open("mentioned_persons_"+year+".pkl",'rb'))
		mentions[year] = persons_mentioned
		persons_mentioned = list(persons_mentioned.values())
		for names in persons_mentioned:
			if names != set():
				all_mentioned_persons = all_mentioned_persons.union(names)
	all_mentioned_persons = list(all_mentioned_persons)
	indices = np.random.choice(range(len(all_mentioned_persons)), size=len(all_mentioned_persons), replace=False)
	person_to_id_map = {p_id: ind for p_id, ind in zip(all_mentioned_persons, indices)}
	feat_mats = {}
	for year, mention in mentions.items():
		mat = lil_matrix((len(mention.keys())-1, len(all_mentioned_persons)))
		for clus, mens in mention.items():
			if clus == -1:
				continue
			for m in list(mens):
				mat[clus, person_to_id_map[m]] += 1

		feat_mats[year] = mat
	pkl.dump(feat_mats, open("cluster_acc_mentions.pkl",'wb'))
	pkl.dump(person_to_id_map, open("person_to_id_map.pkl",'wb'))
	return feat_mats, person_to_id_map


def get_cluster_labels(emb, data_input, year, use_c, big, block_num=20):
	hashtag = data_input['url'].tocsr()
	num_user = hashtag.shape[0]
	hashtag_density = hashtag.sum() / (hashtag.shape[0] * hashtag.shape[1])

	if big:
		print("KMeans...")
		inferred_labs = KMeans(n_clusters=cluster_count[year]).fit_predict(emb)
	else:
		print("HDBSCAN...")
		clusterer = hdbscan.HDBSCAN()
		hdbscan_labels = clusterer.fit_predict(emb)
	
	
	if use_c:
		pkl.dump(hdbscan_labels, open("hdbscan_clusters_with_comm_"+year+".pkl",'wb'))
		pkl.dump(emb, open("final_embeddings_with_comm_"+year+".pkl",'wb'))
	elif big:
		pkl.dump(inferred_labs, open("kmeans_clusters_big_"+year+".pkl",'wb'))
		pkl.dump(emb, open("final_embeddings_big_"+year+".pkl",'wb'))
		return emb, inferred_labs
	else:
		pkl.dump(hdbscan_labels, open("hdbscan_clusters_"+year+".pkl",'wb'))
		pkl.dump(emb, open("final_embeddings_"+year+".pkl",'wb'))

	return emb, hdbscan_labels

## SCG METHOD - HAO'S WORK    
def get_coordinated_groups(data_input, year, use_c, big, emb_dim=100):

	if use_c:
		if os.path.exists("hdbscan_clusters_with_comm_"+year+".pkl"):
			final_embeddings = pkl.load(open("final_embeddings_with_comm_"+year+".pkl",'rb'))
			hdbscan_labels = pkl.load(open("hdbscan_clusters_with_comm_"+year+".pkl",'rb'))
			return final_embeddings, hdbscan_labels
	elif big and os.path.exists("hdbscan_clusters_big_"+year+".pkl"):
		final_embeddings = pkl.load(open("final_embeddings_big_"+year+".pkl",'rb'))
		hdbscan_labels = pkl.load(open("hdbscan_clusters_big_"+year+".pkl",'rb'))
		return final_embeddings, hdbscan_labels
	elif not big and os.path.exists("hdbscan_clusters_"+year+".pkl"):
		final_embeddings = pkl.load(open("final_embeddings_"+year+".pkl",'rb'))
		hdbscan_labels = pkl.load(open("hdbscan_clusters_"+year+".pkl",'rb'))
		return final_embeddings, hdbscan_labels

	svd_embeddings = TruncatedSVD(n_components=emb_dim).fit_transform(data_input['url'])
	
	print("SVD done")
	projected_matrix = data_input['mention'].dot(data_input['mention'].transpose())
	data_input['projection'] = projected_matrix

	print("Projection done")
	final_embeddings = projected_matrix.dot(svd_embeddings)
	return get_cluster_labels(final_embeddings, data_input, year, use_c, big)


def main():
	args = get_args()
	print("Loading data ...\n")
	user_file, url_file, full_df, filtered_df = load_data(args.communities_path, args.full_df_path, args.big_communities)

	clusters_per_year = {}
	full_filtered_df = {}
	group_statistics = {}
	mentioned_persons = {}
	author_ids_per_cluster = {}
	ids_per_cluster = {}

	print("Generating clusters ...\n")
	for year, df in filtered_df.items():
		author_mention_graph, author_url_graph, df = build_graphs(df, year, args.big_communities)

		if args.use_community_info: # if we want to include the community information also in the clustering
			auth_mention_community_matrix, auth_url_community_matrix = get_community_matrix(df) # returns NxC matrices where C=number of communities
			author_mention_graph = lil_matrix(np.hstack((author_mention_graph.todense(), auth_mention_community_matrix)))
			author_url_graph = lil_matrix(np.hstack((author_url_graph.todense(), auth_url_community_matrix)))
		
		node_embeddings, hdbscan_clusters = get_coordinated_groups({'mention':author_mention_graph, 'url':author_url_graph}, year, args.use_community_info, args.big_communities)
		clusters_per_year[year] = hdbscan_clusters
		full_filtered_df[year] = df
		group_statistics[year], mentioned_persons[year] = statistics(df, hdbscan_clusters, year, args.use_community_info, args.big_communities)
		author_ids_per_cluster[year], ids_per_cluster[year] = get_author_info(author_url_graph, hdbscan_clusters, df)

	if args.big_communities:
		pkl.dump(author_ids_per_cluster, open("author_ids_per_cluster_big.pkl",'wb'))
	else:
		pkl.dump(author_ids_per_cluster, open("author_ids_per_cluster.pkl",'wb'))
	get_box_plots(group_statistics,size='small')
	# feat_mats, person_to_id_map = account_mentions()
	# names_in_all_years, clusters_with_common_handles = common_handles(feat_mats, person_to_id_map)
	# get_heatmap(feat_mats, clusters_with_common_handles)


if __name__ == '__main__':
	main()