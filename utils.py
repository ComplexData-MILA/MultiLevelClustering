import pandas as pd
import pickle as pkl
import numpy as np
import argparse
from tqdm import tqdm
import ast
import os
from scipy.sparse import lil_matrix
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

plt.rcParams.update({'font.size': 18})

def get_args():
	parser = argparse.ArgumentParser(
                    prog = 'miltilevel_clustering',
                    description = 'Detects organized groups from twitter',
                    epilog = '')

	parser.add_argument('-com_path', '--communities_path', help='Path to the files containing small communities. Place all community pickle files in a separate folder.', \
		default='small_community_results/')
	parser.add_argument('-full_path', '--full_df_path', help='Path to the full dataframes in a separate folder')
	parser.add_argument('-use_c', '--use_community_info', action='store_true', help='aggregate the embeddings based on community membership or no')
	parser.add_argument('-big', '--big_communities', action='store_true', help='indicates whether we are dealing with large or small communities')
	args = parser.parse_args()

	return args

def load_data(community_path, full_file_path, big=False):
	'''
	This function reads in the community information and keeps only the corresponding author IDs from the full files 
	args: 
			community_path : path to pickle files containing the filtered small community assignments. Both URL and users communities are included separately 
			full_file_path : path to dataframes containing all the data for each of the years separately

	returns:
			user_file 	   : dictionary of author ID to community based on author-user mentions for each year (dict of dict)
			url_file	   : dictionary of author ID to community based on author-url for each year (dict of dict)
			full_df 	   : dictionary of the full dataframe for each year (dict of df)
			filtered_df	   : dictionary of dataframe containing only the authors from the communities for each year (dict of df)
	'''
	community_info_urls = {}
	community_info_user = {}
	full_df = {}
	filtered_df = {} # contains only the nodes from the communities

	if community_path[-1] == '/':
		community_path = community_path[:-1]
	if full_file_path[-1] == '/':
		full_file_path = full_file_path[:-1]

	years = ['2017', '2018', '2019', '2020', '2021'] # adjust this according to your dataset

	for year in years:
		if big:
			url_file = pkl.load(open(community_path+"/mapComm"+year[-2:]+"url100m.pkl",'rb'))
			user_file = pkl.load(open(community_path+"/mapComm"+year[-2:]+"user100m.pkl",'rb'))
		else:
			url_file = pkl.load(open(community_path+"/mapComm"+year[-2:]+"url2_99.pkl",'rb'))
			user_file = pkl.load(open(community_path+"/mapComm"+year[-2:]+"user2_99.pkl",'rb'))

		community_info_urls[year] = url_file
		community_info_user[year] = user_file

		df = pd.read_csv(full_file_path+"/Tweets_"+year+"_20220928.csv",index_col=False)
		full_df[year] = df

		authors_from_url_comms = list(url_file.keys())
		authors_from_user_comms = list(user_file.keys())

		authors_from_all_comms = authors_from_user_comms + authors_from_url_comms
		subgraph_df = df[df['author id'].isin(authors_from_all_comms)]
		subgraph_df['user_community'] = subgraph_df['author id'].apply(lambda x: user_file[x] if x in user_file else -1)
		subgraph_df['url_community'] = subgraph_df['author id'].apply(lambda x: url_file[x] if x in url_file else -1)

		filtered_df[year] = subgraph_df

	return user_file, url_file, full_df, filtered_df


def get_author_info(aut_url_mat, cluster_labels, filtered_df):

	author_cluster_map = defaultdict(list)
	## MAPPING AUTHOR IDS TO AN INDEX FOR THE MATRIX
	id_author_map = {ind: a_id for ind, a_id in zip(range(filtered_df['author id'].nunique()),\
	                                       filtered_df['author id'].unique())}
	inds_per_cluster = defaultdict(list)
	for index, label in enumerate(cluster_labels):
		author_cluster_map[label].append(id_author_map[index])
		inds_per_cluster[label].append(index)

	return author_cluster_map, inds_per_cluster
	


def build_graphs(filtered_df, year, big):
	'''
	This function takes in a dataframe and converts it into two graphs - Author-mention and Author-URL based on the tweets
	args:
			filtered_df  	   : dataframe obtained from load_data containing only those authors from the communities

	returns:
			author_mention_mat : adjacency matrix of author-user mention graph
			author_url_mat	   : adjacency matrix of author-url graph
			filtered_df 	   : returning the input df as we make changes to some of the columns
	'''


	## CONVERTING STRING OF LIST TO LIST - lists in dataframes get saved in a string format
	filtered_df['username_mentioned_id'] = filtered_df['username_mentioned_id'].apply(lambda x: ast.literal_eval(x) \
	                                                                if (not pd.isna(x) and x!=' ') else [])
	filtered_df['username_mentioned_id2'] = filtered_df['username_mentioned_id2'].apply(lambda x: ast.literal_eval(x) \
	                                                                if (not pd.isna(x) and x!=' ') else [])

	## CONSIDERING BOTH USERNAMES MENTIONED IN ORIGINAL TWEET AND IN RETWEETS BY AN AUTHOR
	all_usernames = filtered_df['username_mentioned_id'] + filtered_df['username_mentioned_id2']

	usernames = []
	for item in all_usernames:
	    usernames.extend(item)
	usernames = list(set(usernames))

	## MAPPING AUTHOR IDS TO AN INDEX FOR THE MATRIX
	author_id_map = {a_id: ind for ind, a_id in zip(range(filtered_df['author id'].nunique()),\
	                                       filtered_df['author id'].unique())}

	if big and os.path.exists("mentioned_user_map_big_"+year+".pkl"):
		mentioned_user_map = pkl.load(open("mentioned_user_map_big_"+year+".pkl",'rb'))
	elif not big and os.path.exists("mentioned_user_map_"+year+".pkl"):
		mentioned_user_map = pkl.load(open("mentioned_user_map"+year+".pkl",'rb'))
	else:
		## MAPPING MENTIONED IDS TO MATRIX INDICES
		mentioned_user_map = {}
		dont_use = []
		for item in tqdm(usernames):
		    if item in author_id_map:
		        mentioned_user_map[item] = author_id_map[item]
		        dont_use.append(author_id_map[item])
		    else:
		        options = list(set(list(range(len(usernames)))) - set(dont_use))
		        ind = np.random.choice(options, replace=False)
		        mentioned_user_map[item] = ind
		        dont_use.append(ind)
		if big:
			pkl.dump(mentioned_user_map, open("mentioned_user_map_big_"+year+".pkl",'wb'))
		else:
			pkl.dump(mentioned_user_map, open("mentioned_user_map"+year+".pkl",'wb'))
	filtered_df['author_mat_index'] = filtered_df['author id'].apply(lambda x: author_id_map[x])
	num_authors = filtered_df['author id'].nunique()
	num_uname = max(mentioned_user_map.values())

	if big and os.path.exists("author_mention_matrix_big"+year+".pkl"):
		author_mention_mat = pkl.load(open("author_mention_matrix_big"+year+".pkl",'rb'))
	elif not big and os.path.exists("author_mention_matrix"+year+".pkl"):
		author_mention_mat = pkl.load(open("author_mention_matrix"+year+".pkl",'rb'))
	else:
		author_mention_mat = lil_matrix((num_authors, num_uname+1), dtype=np.int32)

		## BUILDING THE SPARSE MATRIX OF AUTHOR-MENTIONS
		for id, row in tqdm(filtered_df.iterrows()):
		    a_id = row['author_mat_index']
		    mentions = all_usernames.loc[id]
		    for mention in mentions:
		        mention_id = mentioned_user_map[mention]
		        current = author_mention_mat[a_id, mention_id]
		        author_mention_mat[a_id, mention_id] = current + 1

		if big:
			pkl.dump(author_mention_mat, open("author_mention_matrix_big"+year+".pkl",'wb'))
		else:
			pkl.dump(author_mention_mat, open("author_mention_matrix"+year+".pkl",'wb'))


	## CONVERTING STRING OF LIST TO LIST
	filtered_df['urls_expanded'] = filtered_df['urls_expanded'].apply(lambda x: ast.literal_eval(x) \
	                                                                if (not pd.isna(x) and x!=' ') else [])
	filtered_df['urls_expanded2'] = filtered_df['urls_expanded2'].apply(lambda x: ast.literal_eval(x) \
	                                                                if (not pd.isna(x) and x!=' ') else [])
	## CONSIDERING BOTH USERNAMES MENTIONED IN ORIGINAL TWEET AND IN RETWEETS BY AN AUTHOR
	all_urls = filtered_df['urls_expanded'] + filtered_df['urls_expanded2']
	urls = []
	for item in all_urls:
	    urls.extend(item)
	urls = list(set(urls))

	## MAPPING MENTIONED IDS TO MATRIX INDICES
	url_map = {}
	for id, item in tqdm(enumerate(urls)):
	    url_map[item] = id

	if big and os.path.exists("author_url_matrix_big"+year+".pkl"):
		author_url_mat = pkl.load(open("author_url_matrix_big"+year+".pkl",'rb'))
	elif not big and os.path.exists("author_url_matrix"+year+".pkl"):
		author_url_mat = pkl.load(open("author_url_matrix"+year+".pkl",'rb'))
	else:
		## BUILDING THE SPARSE MATRIX OF AUTHOR-URLs
		author_url_mat = lil_matrix((num_authors, max(url_map.values())+1), dtype=np.int32)

		for id, row in tqdm(filtered_df.iterrows()):
		    a_id = row['author_mat_index']
		    u_list = all_urls.loc[id]
		    for u in u_list:
		        url_id = url_map[u]
		        current = author_url_mat[a_id, url_id]
		        author_url_mat[a_id, url_id] = current + 1

		if big:
			pkl.dump(author_url_mat, open("author_url_matrix_big"+year+".pkl",'wb'))
		else:
			pkl.dump(author_url_mat, open("author_url_matrix"+year+".pkl",'wb'))

	return author_mention_mat, author_url_mat, filtered_df


def statistics(filtered_df, inferred_labels, year, use_c, big):
	'''
	This function returns the group statistics of clusters (number of authors, tweets, etc) for the boxplots from the paper
	args:	  
			filtered_df 	  : dataframe containing author ids, tweets, etc for which we have inferred labels
			inferred_labels   : HDBSCAN labels for all the authors in filtered_df
 
	returns:
			grouping_df		  : dataframe containing the statistics for each cluster
			mentioned_persons : dictionary of handles (twitter, onlyfans, etc) extracted from the URLs of the tweets
	'''

	# if use_c:
	# 	if os.path.exists("grouping_df_with_comm_"+year+".pkl"):
	# 		grouping_df = pkl.load(open("grouping_df_with_comm_"+year+".pkl",'rb'))
	# 		mentioned_persons = pkl.load(open("mentioned_persons_with_comm_"+year+".pkl",'rb'))
	# 		return grouping_df, mentioned_persons
	# elif os.path.exists("grouping_df_"+year+".pkl"):
	# 	grouping_df = pkl.load(open("grouping_df_"+year+".pkl",'rb'))
	# 	mentioned_persons = pkl.load(open("mentioned_persons_"+year+".pkl",'rb'))
	# 	return grouping_df, mentioned_persons

	id_author_map = {ind: a_id for ind, a_id in zip(range(filtered_df['author id'].nunique()),\
	                                       filtered_df['author id'].unique())}
	author_cluster_label = defaultdict(list)
	for i, label in enumerate(inferred_labels):
	    author_cluster_label[label].append(id_author_map[i])

	filtered_df.rename(columns={'id':'tweet_id'},inplace=True)
	grouping_df = pd.DataFrame()
	grp_ids = []
	num_authors = []
	num_of = []
	num_urls = []
	num_twitter = []
	num_tweets = []
	# num_sources = []
	num_retweets = []
	num_users = []
	of_entropy = []
	mentioned_persons = {}

	for cluster in tqdm(author_cluster_label):
	    auts = author_cluster_label[cluster]
	    grp = filtered_df[filtered_df['author id'].isin(auts)]
	    grp_ids.append(cluster)
	    num_authors.append(grp['author id'].nunique())
	    num_tweets.append(grp['tweet_id'].nunique())
	    # num_sources.append(len(list(set(grp['source'].values) & set(uncommon_sources))))
	    num_retweets.append((grp[grp['Retweet']=="'retweeted"].tweet_id.nunique()/num_tweets[-1])*100)
	    
	    all_urls = []
	    ofs = set()
	    of_counts = defaultdict(int)
	    twitters = set()
	    for url in grp['urls_expanded']:
	        all_urls.extend(url)
	        for u in url:
	            u = u.lower()
	            if 'onlyfans' in u:
	                if len(u.split('.com/')) < 2:
	                    continue
	                person_mentioned = u.split(".com/")[1]
	                if '/' in person_mentioned:
	                    person_mentioned = person_mentioned.split('/')[0]
	                if person_mentioned.isnumeric():
	                    continue
	                if 'ref' in person_mentioned:
	                    person_mentioned = person_mentioned.split('?ref')[0]
	                ofs.add(person_mentioned)
	                of_counts[person_mentioned] += 1
	            elif 'twitter' in u:
	                if len(u.split('.com/')) < 2:
	                    continue
	                person_mentioned = u.split(".com/")[1]
	                if '/' in person_mentioned:
	                    person_mentioned = person_mentioned.split('/')[0]
	                if person_mentioned.isnumeric():
	                    continue
	                if 'ref' in person_mentioned:
	                    person_mentioned = person_mentioned.split('?ref')[0]
	                twitters.add(person_mentioned)
	    for url in grp['urls_expanded2']:
	        all_urls.extend(url)
	        for u in url:
	            u = u.lower()
	            break
	            if 'onlyfans' in u:
	                if len(u.split('.com/')) < 2:
	                    continue
	                person_mentioned = u.split(".com/")[1]
	                if '/' in person_mentioned:
	                    person_mentioned = person_mentioned.split('/')[0]
	                if person_mentioned.isnumeric():
	                    print(person_mentioned)
	                    continue
	                if 'ref' in person_mentioned:
	                    person_mentioned = person_mentioned.split('?ref')[0]
	                ofs.add(person_mentioned)
	                of_counts[person_mentioned] += 1
	            elif 'twitter' in u:
	                if len(u.split('.com/')) < 2:
	                    continue
	                person_mentioned = u.split(".com/")[1]
	                if '/' in person_mentioned:
	                    person_mentioned = person_mentioned.split('/')[0]
	                if person_mentioned.isnumeric():
	                    continue
	                if 'ref' in person_mentioned:
	                    person_mentioned = person_mentioned.split('?ref')[0]
	                twitters.add(person_mentioned)
	    pms = ofs.union(twitters)
	    num_urls.append(len(set(all_urls)))
	    num_of.append(len(ofs))
	    num_twitter.append(len(twitters))
	    mentioned_persons[cluster] = pms
	    num_users.append(len(pms))
	    of_entropy.append(entropy(list(of_counts.values()))*len(of_counts))

	grouping_df['cluster_id'] = grp_ids
	grouping_df['num_urls'] = num_urls
	grouping_df['num_tweets'] = num_tweets
	grouping_df['num_retweets'] = num_retweets
	grouping_df['num_twitter'] = num_twitter
	grouping_df['num_OF'] = num_of
	grouping_df['num_authors'] = num_authors
	grouping_df['num_users'] = num_users
	# grouping_df['num_sources'] = num_sources
	grouping_df['of_entropy'] = of_entropy

	if big:
		pkl.dump(grouping_df, open("grouping_df_big_"+year+".pkl",'wb'))
		pkl.dump(mentioned_persons, open("mentioned_persons_big_"+year+".pkl",'wb'))
	else:
		pkl.dump(grouping_df, open("grouping_df_"+year+".pkl",'wb'))
		pkl.dump(mentioned_persons, open("mentioned_persons_"+year+".pkl",'wb'))
	return grouping_df, mentioned_persons

def get_community_matrix(filtered_df):
	'''
	This function returns an NxC matrix each for the Author-mention and Author-URL graphs where we have their community membership information

	args:
			filtered_df	  : The dataframe containing the author-ids, author indices and community memberships

	returns:
			user_comm_mat : NxC matrix where the author-mention community membership is represented as a 1-hot encoding for each of the N author ids
			url_comm_mat  : NxC matrix where the author-url community membership is represented as a 1-hot encoding for each of the N author ids
	'''
	num_authors = filtered_df['author id'].nunique()
	user_comm_mat = np.zeros([num_authors, filtered_df["user_community"].nunique()])
	url_comm_mat = np.zeros([num_authors, filtered_df['url_community'].nunique()])

	user_comm_id_map = {a_id: ind for ind, a_id in zip(range(filtered_df['user_community'].nunique()),\
	                                       filtered_df['user_community'].unique())}
	url_comm_id_map = {a_id: ind for ind, a_id in zip(range(filtered_df['url_community'].nunique()),\
	                                       filtered_df['url_community'].unique())}
	
	for id, df_row in filtered_df[['author_mat_index', 'user_community', 'url_community']].iterrows():
		row = df_row.author_mat_index
		col_user = user_comm_id_map[df_row.user_community]
		col_url = url_comm_id_map[df_row.url_community]

		user_comm_mat[row, col_user] = 1
		url_comm_mat[row, col_url] = 1

	return user_comm_mat, url_comm_mat

def get_heatmap(full_name_feats, clusters_with_common_handles):
	figs, ax = plt.subplots(nrows=1, ncols=6, figsize=[12,5], gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
	ax[0].get_shared_y_axes().join(ax[1],ax[2],ax[3],ax[4])
	# ax[0].get_shared_y_axes().join(ax)
	for i, year in enumerate(list(full_name_feats.keys())):
	    df = full_name_feats[year].todense().transpose()
	    clusters_to_keep = clusters_with_common_handles[year]
	    df = df[clusters_to_keep]
	    if year =='2021':
	        p1 = sns.heatmap(df,ax=ax[i],cmap='binary_r', cbar_ax=ax[-1])
	    else:
	        p1 = sns.heatmap(df,ax=ax[i],cmap='binary_r', cbar=False)
	    ax[i].set_ylabel('')
	    ax[i].set_xlabel('')
	    ax[i].set_title(year)
	    if i > 0:
	        ax[i].set_yticks([])
	# figs.supxlabel("Cluster ID")
	# figs.supylabel("Handle ID")
	# plt.savefig("cluster_handle.png",bbox_inches='tight')
	plt.show()

def get_box_plots(grouping_df, size='big'):
    nrows=2
    ncols=4
    figs, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[24,12])
    all_cols = list(set(grouping_df['2017'].columns)-{'cluster_id'})
    curr_col = 0
    all_bplots = []
    columns = ['URLs', 'Tweets', 'Retweets', 'Twitter accounts', 'OnlyFans accounts', 'Authors','User mentions', 'OnlyFans acc. entropy']
    for i in range(nrows):
        for j in range(ncols):
            all_bplots.append(ax[i,j].boxplot([np.log(df[all_cols[curr_col]]+1) for k, df in grouping_df.items()],\
                            showmeans=True, meanline=True,vert=True, patch_artist=True,\
                            meanprops=dict(color='red', linestyle='-',linewidth=2.9), \
                            medianprops=dict(color='green', linestyle='--',linewidth=0.9)))
            ax[i,j].set_xticks(ticks=range(1,6), labels=list(grouping_df.keys()), rotation=45, ha='center')
            ax[i,j].set_title(columns[curr_col])
            curr_col += 1
            if curr_col >= len(all_cols):
                rem_i = nrows-i
                rem_j = ncols-j
                for k in range(1,rem_j):
                    ax[i,j+k].set_visible(False)
                break
    # fill with colors
    for bplot in all_bplots:
        for patch in bplot['boxes']:
            patch.set(alpha=0.2,linewidth=0.9,edgecolor='k')
#     figs.supylabel("log_e (count+1)")
    plt.subplots_adjust(top = 0.79, bottom=0.21, hspace=0.5, wspace=0.2, left=0.09)
    plt.show()
