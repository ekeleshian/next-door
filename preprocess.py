import pickle
import json
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import config


def preprocess_text(text):
    list_of_text = text.split('\n')
    clean_list = list_of_text[1:3]
    for row in list_of_text[3:]:
        words = row.split(' ')
        if len(words) > 7:
            clean_list.append(row)
    return ' '.join(clean_list)


def lda_serialize(text):
    count = CountVectorizer(stop_words='english', max_df=config.MAX_DF, max_features=config.MAX_FEATURES)
    X = count.fit_transform(text)
    lda = LatentDirichletAllocation(n_components=config.N_TOPICS, random_state=123, learning_method='batch', batch_size=config.BATCH_SIZE)
    X_topics = lda.fit_transform(X)
    print('fit and transform executed')
    with open(f'count-max_df={config.MAX_DF}-max_features={config.MAX_FEATURES}-ntopics={config.N_TOPICS}-batch_size={config.BATCH_SIZE}.pkl', 'wb') as f:
        pickle.dump(count, f)
    with open(f'topics-max_df={config.MAX_DF}-max_features={config.MAX_FEATURES}-ntopics={config.N_TOPICS}-batch_size={config.BATCH_SIZE}.pkl', 'wb') as f:
        pickle.dump(X_topics, f)
    with open(f'lda-max_df={config.MAX_DF}-max_features={config.MAX_FEATURES}-ntopics={config.N_TOPICS}-batch_size={config.BATCH_SIZE}.pkl', 'wb') as f:
        pickle.dump(lda, f)


def get_path_idx(count_paths, topic_paths, lda_paths):
    topic_chopped = [path[len('topics'):] for path in topic_paths]
    lda_chopped = [path[len('lda'):] for path in lda_paths]
    triplets = []
    for idx, path in enumerate(count_paths):
        short_path = path[len('count'):]
        triplets.append((idx, topic_chopped.index(short_path), lda_chopped.index(short_path)))
    return triplets


def analyze_all_models(clean_text):
    count_paths = glob("count*pkl")
    topic_paths = glob('topic*pkl')
    lda_paths = glob('lda*pkl')
    path_triplets = get_path_idx(count_paths, topic_paths, lda_paths)
    all_results_dict = dict()
    for triplet in path_triplets:
        count_idx, topic_idx, lda_idx = triplet
        count_path = count_paths[count_idx]
        topic_path = topic_paths[topic_idx]
        lda_path = lda_paths[lda_idx]
        results_dict = jsonify_results(clean_text, count_path, topic_path, lda_path)
        hyperparameters = count_path[len('count-'):]
        print(f'appending model: {hyperparameters} in dict')
        all_results_dict[hyperparameters] = results_dict

    with open('all_results.json', 'w') as json_file:
        json.dump(all_results_dict, json_file)
        print('saved all results in all_results.json')


def jsonify_results(clean_text, count_path, topic_path, lda_path):
    results_dict = dict()
    with open(count_path, 'rb') as f:
        count = pickle.load(f)
    with open(lda_path, 'rb') as fi:
        lda = pickle.load(fi)
    with open(topic_path, 'rb') as fil:
        X_topics = pickle.load(fil)
    n_top_words = 20
    feature_names = count.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        sub_results = results_dict[f"topic_{topic_idx + 1}"] = dict()
        sub_results["lda-components"] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]])
        sub_results['X_topic'] = posts = []
        x_topic = X_topics[:, topic_idx].argsort()[::-1]
        for iter_idx, post_idx in enumerate(x_topic[:20]):
            posts.append(clean_text[post_idx])
    return results_dict


if __name__ == "__main__":
    with open('bag-of-text.pkl', 'rb') as f:
        bag_of_text = pickle.load(f)

    clean_text = list(map(preprocess_text, bag_of_text))
    # lda_serialize(clean_text)
    # print('pickle dump done')
    analyze_all_models(clean_text)
