#!/usr/bin/env python
# coding: utf-8

'''
Authors : Junhao Wang and Pratheeksha Nair

This file contains code for reproducing results on the synthetic data experiments. It first generates the synthetic data if it doesn't already exist
and then runs all the baselines.

To run:

python Baselines.py

'''

import warnings

warnings.filterwarnings("ignore")

# # Gen data

import time
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from tqdm.autonotebook import tqdm
from scipy.sparse import diags, csr_matrix, hstack
from sklearn.metrics import f1_score
import numpy as np
import itertools
from copy import deepcopy
from functools import reduce
import hdbscan

import sparse
import tensorly as tl
from tensorly import tensor as tensor_dense
from tensorly import unfold as unfold_dense
from tensorly.contrib.sparse import tensor, unfold
from tensorly.tenalg import inner, mode_dot, multi_mode_dot

from random import sample
import warnings
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import collections
from collections import defaultdict

import community.community_louvain as community
import networkx as nx

KMeans = MiniBatchKMeans

def label2indexset(labels):
	indexsets = defaultdict(list)
	indexsets_ = {}
	for ind, i in enumerate(labels):
		indexsets[i].append(ind)
	for i in indexsets:
		indexsets_[i] = set(indexsets[i])
	return indexsets_


def quality_measure(labels, inferred_labels):
	labels_indexsets = label2indexset(labels)
	inferred_labels_indexsets = label2indexset(inferred_labels)
	q_sum = 0
	k = len(labels_indexsets)
	for i in labels_indexsets:
		min_j = 0
		set_i = labels_indexsets[i]
		for j in inferred_labels_indexsets:
			set_j = inferred_labels_indexsets[j]
			j_sim = len(set_j.intersection(set_i)) / len(set_j.union(set_i))
			if j_sim > min_j:
				min_j = j_sim
		q_sum += min_j
	return q_sum / k


def get_random_samples_higherorder(ll, p):
	assert p <= 1
	ll_len = list(map(lambda x: len(x), ll))
	n_prod = int(reduce(lambda a, b: a * b, ll_len) * p)
	#     print('sample {} from {}'.format(n_prod, [len(i) for i in ll]))
	edge_indices = []
	for l in ll:
		edge_indices.append(np.array(l)[np.random.randint(len(l), size=n_prod)].tolist())
	return edge_indices


def gen_cluster(block_sizes, dim_sizes, bipartite, ps, q):
	print('check valid')
	# check valid
	if not bipartite:
		for i in block_sizes:
			assert i[-1] == i[-2]
		assert dim_sizes[-1] == dim_sizes[-2]

	assert np.array(block_sizes).shape[1] == len(dim_sizes)
	assert (np.array(block_sizes).sum(axis=0) <= np.array(dim_sizes)).all()

	assert 1 >= max(ps) >= min(ps) > q >= 0

	back_size = deepcopy(dim_sizes)
	for i in block_sizes:
		for ind, j in enumerate(i):
			back_size[ind] -= j

	cluster_id2idx = {}
	c = 0
	ind = -1
	# for ind, i in enumerate(block_sizes + [back_size]):
	for i in block_sizes:
		ind += 1
		cluster_id2idx[ind] = []
		for ind_j, j in enumerate(i):
			if ind_j == len(i) - 2:
				cluster_id2idx[ind].append(list(range(c, c + j)))
				c += j
			elif ind_j == len(i) - 1:
				if bipartite:
					cluster_id2idx[ind].append(sample(range(dim_sizes[ind_j]), j))
				else:
					cluster_id2idx[ind].append(deepcopy(cluster_id2idx[ind][-1]))
			else:
				cluster_id2idx[ind].append(sample(range(dim_sizes[ind_j]), j))


	all_edge_indices = []
	c = -1
	for i in cluster_id2idx:
		c += 1
		ll = cluster_id2idx[i]
		p = ps[c]
		all_edge_indices.append(get_random_samples_higherorder(ll, p))
		if not bipartite and len(dim_sizes) == 2:
			# add dense inter group connection

			all_edge_indices.append(get_random_samples_higherorder([ll[0], ll[1]], p / 2))

			# add semi dense group background connection

			all_edge_indices.append(get_random_samples_higherorder([ll[0], list(range(dim_sizes[1]))], p / 4))
			all_edge_indices.append(get_random_samples_higherorder([list(range(dim_sizes[0])), ll[1]], p / 4))

	all_edge_indices.append(get_random_samples_higherorder(list(map(lambda x: list(range(x)), dim_sizes)), q))

	all_edge_indices_ = [[] for _ in range(len(dim_sizes))]
	for ei in tqdm(all_edge_indices):
		for ind, ei_ in enumerate(ei):
			all_edge_indices_[ind] += ei_

	return {
		'cluster_id2idx': cluster_id2idx,
		'edge_indices': all_edge_indices,
		'tensor': gen_tensor(all_edge_indices_, dim_sizes)
	}


def gen_tensor(all_edge_indices, dim_sizes):
	data = [1.] * len(all_edge_indices[0])
	s = sparse.COO(all_edge_indices, data, shape=tuple(dim_sizes)).astype(bool).astype(int)
	return tensor(s)


if __name__ == '__main__':

	all_perform = {}
	all_perform_time = {}

	normalized_mutual_info_score = quality_measure

	for num_user in [2000, 6000, 10000, 14000, 18000, 22000, 26000, 30000]:

		all_perform[num_user] = {}
		all_perform[num_user]['mlc'] = []
		all_perform[num_user]['strict'] = []
		all_perform[num_user]['sgp'] = []
		all_perform[num_user]['sgp_self'] = []
		all_perform[num_user]['sgp_dot'] = []
		all_perform[num_user]['sgp_dot_self'] = []
		all_perform[num_user]['louvain'] = []
		all_perform[num_user]['infomap'] = []
		all_perform[num_user]['fraudar'] = []
		all_perform[num_user]['node2vec'] = []
		all_perform[num_user]['attri2vec'] = []
		all_perform[num_user]['graphsage'] = []
		all_perform[num_user]['tiny'] = []

		all_perform_time[num_user] = {}
		all_perform_time[num_user]['mlc'] = []
		all_perform_time[num_user]['strict'] = []
		all_perform_time[num_user]['sgp'] = []
		all_perform_time[num_user]['sgp_self'] = []
		all_perform_time[num_user]['sgp_dot'] = []
		all_perform_time[num_user]['sgp_dot_self'] = []
		all_perform_time[num_user]['louvain'] = []
		all_perform_time[num_user]['infomap'] = []
		all_perform_time[num_user]['fraudar'] = []
		all_perform_time[num_user]['node2vec'] = []
		all_perform_time[num_user]['attri2vec'] = []
		all_perform_time[num_user]['graphsage'] = []
		all_perform_time[num_user]['tiny'] = []

		print('============= num user: ', num_user)
		repeat = 1 # CHANGE THIS TO REPEAT THE EXPERIMENTS MULTIPLE TIME

		for meta_run in range(repeat):

			print('==== meta run', meta_run)

			num_hashtag = num_user

			block_size = 20
			block_num = 8

			p = .02
			q = .005

			s_h = 80
			s_l = 10
			t_h = 80
			t_l = 10
			min_p = .01
			min_q = .01
			emb_dim = 10

			if os.path.exists('data_input_{}_{}.pkl'.format(num_user, meta_run)):
				data_input = pickle.load(open('data_input_{}_{}.pkl'.format(num_user, meta_run), 'rb'))
				labs = data_input['labs']
				labs_binary = data_input['labs_binary']
			else:
				# SYNTHETIC DATA GENERATION PROCESS
				config = {
					'follow': {
						'block_sizes': [[block_size, block_size] for _ in range(block_num)],
						'dim_sizes': [num_user, num_user],
						'bipartite': False,
						'ps': [p] * block_num,  # [.4,.3,.2, .1, .06],
						'q': q
					},
					'hashtag': {
						'block_sizes': [[block_size, block_size] for _ in range(block_num)],
						'dim_sizes': [num_user, num_hashtag],
						'bipartite': True,
						'ps': [p] * block_num,
						'q': q
					}
				}

				for i in config:
					if not config[i]['bipartite']:
						for j in config[i]['block_sizes']:
							assert j[-1] == j[-2]

				cluster_info = {}

				for i in config:
					print(i)
					cluster_info[i] = gen_cluster(**config[i])
					print('\n')

				data_config = {
					'follow': {'axis': ['user', 'user'], 'relweight': 1},
					'hashtag': {'axis': ['user', 'hashtag'], 'relweight': 1},
				}

				data_input = {i: cluster_info[i]['tensor'] for i in data_config}

				labs = np.array([0] * cluster_info['follow']['tensor'].shape[0])
				for i in cluster_info['follow']['cluster_id2idx']:
				   labs[np.array(cluster_info['follow']['cluster_id2idx'][i][0])] = i + 1

				labs_binary = (labs > 0).astype(int)

				data_input['follow'] = data_input['follow'].astype(np.int16)
				data_input['hashtag'] = data_input['hashtag'].astype(np.int16)
				data_input['labs'] = labs
				data_input['labs_binary'] = labs_binary

				pickle.dump(data_input, open('data_input_{}_{}.pkl'.format(num_user, meta_run), 'wb'))


			def evaluate_emb(emb):

				inferred_labs = KMeans(n_clusters=block_num + 1).fit_predict(emb)

				inferred_labs_binary = np.zeros(num_user)

				for i in set(inferred_labs):
					c_idx = np.argwhere(inferred_labs == i).reshape(-1)
					c_p = data_input['follow'].to_scipy_sparse().tocsr()[c_idx][:, c_idx].sum() / (len(c_idx) ** 2)
					if c_p >= min_p:
						inferred_labs_binary[c_idx] = 1

				return (normalized_mutual_info_score(labs, inferred_labs),
						f1_score(labs_binary, inferred_labs_binary)), emb


			def evaluate_emb_allcheck(emb, data_input):
				hashtag = data_input['hashtag'].to_scipy_sparse().tocsr()
				hashtag_density = hashtag.sum() / (hashtag.shape[0] * hashtag.shape[1])

				inferred_labs = KMeans(n_clusters=block_num + 1).fit_predict(emb)

				inferred_labs_binary = np.zeros(num_user)

				inferred_labs_binary_allcheck = np.zeros(num_user)

				inferred_labs_binary_partcheck = np.zeros(num_user)

				for i in set(inferred_labs):
					c_idx = np.argwhere(inferred_labs == i).reshape(-1)
					c_p = data_input['follow'].to_scipy_sparse().tocsr()[c_idx][:, c_idx].sum() / (len(c_idx) ** 2)
					if c_p >= min_p:
						inferred_labs_binary[c_idx] = 1

						if len(c_idx) >= s_l and len(c_idx) <= s_h:
							inferred_labs_binary_partcheck[c_idx] = 1
							#                 print('all check row size ok')
							row_sum = np.asarray(hashtag[c_idx].sum(axis=0)).reshape(-1)
							c_idx_col = np.argwhere(
								(min_p / hashtag_density) ** row_sum * ((1 - min_p) / (1 - hashtag_density)) ** (
										len(c_idx) - row_sum) >= 1
							).reshape(-1)
							#                 print('all check column size', len(c_idx_col))
							if len(c_idx_col) >= t_l and len(c_idx_col) <= t_h:
								#                     print('all check column size ok')
								c_p_col = hashtag[c_idx][:, c_idx_col].sum() / (len(c_idx) * len(c_idx_col))
								if c_p_col >= min_q:
									#                         print('all check condidiont true')
									inferred_labs_binary_allcheck[c_idx] = 1

				return (normalized_mutual_info_score(labs, inferred_labs), max(
					[f1_score(labs_binary, inferred_labs_binary),
					 f1_score(labs_binary, inferred_labs_binary_allcheck),
					 f1_score(labs_binary, inferred_labs_binary_partcheck)
					 ]
				)), emb, inferred_labs

			# strict intersection (Maricarmen's method)
			def strict_intersection_louvain(data_input):
				proj1 = data_input['follow'].dot(data_input['follow'].transpose())
				proj2 = data_input['follow'].dot(data_input['hashtag'].transpose())
				G1 = nx.from_scipy_sparse_matrix(proj1.tocsr())
				G2 = nx.from_scipy_sparse_matrix(proj2.tocsr())
				# first compute the best partition
				partition1 = community.best_partition(G1)
				partition2 = community.best_partition(G2)

				partition_node1 = defaultdict(list)
				partition_node2 = defaultdict(list)

				for n, p in partition1.items():
					partition_node1[p].append(n)

				for n,p in partition2.items():
					partition_node2[p].append(n)

				intersecting_grps = {}
				g = 0
				clustered_nodes = []
				final_labels = np.zeros(len(G1.nodes))
				for grp1 in tqdm(partition_node1):
					for grp2 in partition_node2:
						intersect = list(set(partition_node1[grp1]) & set(partition_node2[grp2]))
						# print(len(intersect), int(0.4*len(partition_node1[grp1])),int(0.4*len(partition_node2[grp2])))
						# if len(intersect) > 0 and len(intersect) > int(0.4*len(partition_node1[grp1])) and len(intersect) > int(0.4*len(partition_node2[grp2])):
						if len(intersect) > 1:
							intersecting_grps[0] = intersect
							clustered_nodes.extend(intersect)
							for inter in intersect:
								final_labels[inter] = g
							g += 1
				intersecting_grps[-1] = list(set(list(G1.nodes)) - set(clustered_nodes))
				intersecting_grps[-1].extend(list(set(list(G2.nodes)) - set(clustered_nodes)))
				
				return normalized_mutual_info_score(labs, final_labels), [], final_labels

			
			print('strict intersection between clusters\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['strict'].append(strict_intersection_louvain(data_input))
				end = time.time()
				all_perform_time[num_user]['strict'].append(end - start)

			## Joint clustering
			def joint_clustering(data_input):
			    projected_matrix = data_input['follow'].to_scipy_sparse().tocsr().dot(data_input['follow'].to_scipy_sparse().tocsr().transpose())
			    data_input['projection'] = projected_matrix
			    print("Projection done")

			    svd_embeddings = TruncatedSVD(n_components=emb_dim).fit_transform(data_input['hashtag'].to_scipy_sparse().tocsr())    
			    print("SVD done")
		
			    final_embeddings = projected_matrix.dot(svd_embeddings)
			    hashtag = data_input['hashtag'].to_scipy_sparse().tocsr()
			    num_user = hashtag.shape[0]
			    hashtag_density = hashtag.sum() / (hashtag.shape[0] * hashtag.shape[1])

			    print("HDBSCAN...")
			    clusterer = hdbscan.HDBSCAN()
			    hdbscan_labels = clusterer.fit_predict(final_embeddings)

			    return (normalized_mutual_info_score(labs, hdbscan_labels)), final_embeddings, hdbscan_labels


			print('multilevel clustering\n')

			for _ in tqdm(range(repeat)):
			    start = time.time()
			    all_perform[num_user]['mlc'].append(joint_clustering(data_input))
			    end = time.time()
			    all_perform_time[num_user]['mlc'].append(end - start)

			
			# Fraudar

			from baselines.fraudar.greedy import logWeightedAveDegree, detectMultiple

			def eval_fraudar(data_input):

				detected_blocks = logWeightedAveDegree(data_input['follow'].to_scipy_sparse())

				inferred_labs_binary = np.zeros(num_user)
				inferred_labs_binary[np.array(list(detected_blocks[0][0]))] = 1

				return (f1_score(labs_binary, inferred_labs_binary),), inferred_labs_binary

			print('fraudar\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['fraudar'].append(eval_fraudar(data_input))
				end = time.time()
				all_perform_time[num_user]['fraudar'].append(end - start)


			# # tiny

			def eval_tiny(data_input):
				emb = TruncatedSVD(n_components=emb_dim).fit_transform(data_input['hashtag'].to_scipy_sparse())

				return evaluate_emb_allcheck(emb, data_input)

			print('tiny\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['tiny'].append(eval_tiny(data_input))
				end = time.time()
				all_perform_time[num_user]['tiny'].append(end - start)

			# Infomap

			import igraph as ig

			def eval_infomap(data_input):
				g_ig = ig.Graph.Adjacency((data_input['follow'].todense() > 0).tolist(),
										  mode=ig.ADJ_UNDIRECTED)  # convert via adjacency matrix
				c_infomap = g_ig.community_infomap()
				inferred_labs = np.array(c_infomap.membership)

				inferred_labs_binary = np.zeros(num_user)

				for i in set(inferred_labs):
					c_idx = np.argwhere(inferred_labs == i).reshape(-1)
					c_p = data_input['follow'].to_scipy_sparse().tocsr()[c_idx][:, c_idx].sum() / (len(c_idx) ** 2)
					if c_p >= min_p:
						inferred_labs_binary[c_idx] = 1

				return (normalized_mutual_info_score(labs, inferred_labs), f1_score(labs_binary,
																				   inferred_labs_binary)), inferred_labs

			print('infomap\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['infomap'].append(eval_infomap(data_input))
				end = time.time()
				all_perform_time[num_user]['infomap'].append(end - start)

			# Louvain

			import community
			import community.community_louvain as community
			import networkx as nx
			import matplotlib.pyplot as plt


			def eval_louvain(data_input):
				G = nx.from_scipy_sparse_matrix(data_input['follow'].to_scipy_sparse())

				# first compute the best partition
				partition = community.best_partition(G)

				inferred_labs = np.zeros(num_user)

				for i in partition:
					inferred_labs[i] = partition[i]

				inferred_labs_binary = np.zeros(num_user)

				for i in set(inferred_labs):
					c_idx = np.argwhere(inferred_labs == i).reshape(-1)
					c_p = data_input['follow'].to_scipy_sparse().tocsr()[c_idx][:, c_idx].sum() / (len(c_idx) ** 2)
					if c_p >= min_p:
						inferred_labs_binary[c_idx] = 1

				return (normalized_mutual_info_score(labs, inferred_labs), f1_score(labs_binary,
																				   inferred_labs_binary)), inferred_labs

			print('louvain\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['louvain'].append(eval_louvain(data_input))

				end = time.time()
				all_perform_time[num_user]['louvain'].append(end - start)


			# Node2vec

			from baselines.graph_embeddings.node2vec import node2vec

			def eval_node2vec(data_input):
				emb = node2vec(ajacency=data_input['follow'].to_scipy_sparse())
				return evaluate_emb_allcheck(emb, data_input)

			print('node2vec\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['node2vec'].append(eval_node2vec(data_input))

				end = time.time()
				all_perform_time[num_user]['node2vec'].append(end - start)

			# Attri2vec

			from baselines.graph_embeddings.attri2vec import attri2vec

			def eval_attri2vec(data_input):
				emb = attri2vec(data_input['follow'].to_scipy_sparse(), data_input['hashtag'].to_scipy_sparse())
				return evaluate_emb_allcheck(emb, data_input)

			print('attri2vec\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['attri2vec'].append(eval_attri2vec(data_input))
				end = time.time()
				all_perform_time[num_user]['attri2vec'].append(end - start)

			# GraphSage

			from baselines.graph_embeddings.graphsage import graphsage

			def eval_graphsage(data_input):
				emb = graphsage(data_input['follow'].to_scipy_sparse(), data_input['hashtag'].to_scipy_sparse())
				return evaluate_emb_allcheck(emb, data_input)

			print('graphsage\n')

			for _ in tqdm(range(repeat)):
				start = time.time()
				all_perform[num_user]['graphsage'].append(eval_graphsage(data_input))
				end = time.time()
				all_perform_time[num_user]['graphsage'].append(end - start)

			pickle.dump(all_perform, open('all_perform_strict.pkl', 'wb'))
			pickle.dump(all_perform_time, open('all_perform_time_strict.pkl', 'wb'))
			

	pickle.dump(all_perform, open('all_perform_final_strict.pkl', 'wb'))
	pickle.dump(all_perform_time, open('all_perform_time_final_strict.pkl', 'wb'))
