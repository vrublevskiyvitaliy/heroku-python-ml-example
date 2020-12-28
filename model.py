# All imports and installs should be here
import sys

import pandas as pd
import numpy as np
import math
import spacy


# Space module import
# NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
import networkx as nx
from networkx import __version__ as nxv

# linear_sum_assignment Hungarian algorithm
from scipy.optimize import linear_sum_assignment

from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.tokens import Token as SpacyToken


class TfIdf:
    def __init__(self, data):
        self.data = data
        self.prepare_corpus()
        self.fit()

    def prepare_corpus(self):
        self.corpus = [x['s1'] for x in self.data] + [x['s2'] for x in self.data]
        self.corpus = list(set(self.corpus))
        self.corpus = sorted(self.corpus)
        self.corpus_len = len(self.corpus)

        self.sent_to_index = {}
        for index, s in enumerate(self.corpus):
            self.sent_to_index[s] = index

    def fit(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.corpus)

        self.vectorizer = vectorizer

        self.words_list = vectorizer.get_feature_names()
        self.idf = vectorizer._tfidf.idf_

        self.word_to_index = {}
        for index, w in enumerate(self.words_list):
            self.word_to_index[w] = index

    def use_idf(self, t):
        return (t.is_alpha and
                not (t.is_space or t.is_punct or
                     t.is_stop or t.like_num))

    def get_idf(self, token):
        if not self.use_idf(token):
            return 1
        return self.get_word_idf(token.text)

    def get_word_idf(self, word):
        if word in self.word_to_index:
            idf = self.idf[self.word_to_index[word]]
        else:
            # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/feature_extraction/text.py#L1443
            idf = np.log(self.corpus_len + 1 / 1) + 1
        #         print("Calling idf for " + word + " = " + str(idf))
        return idf


def get_data_location():
    return "./dataset/"


def add_start_end_sentence_tokens(s):
    return "%s %s %s" % (SENTENCE_START_TOKEN, s, SENTENCE_END_TOKEN)


def load_data(_preprocess_sentence=None, _train=False, _test=False):
    "Load the MSRP dataset."
    loc = get_data_location()
    trainloc = loc + 'msr_paraphrase_train.txt'
    testloc = loc + 'msr_paraphrase_test.txt'

    if _preprocess_sentence is None:
        _preprocess_sentence = lambda x: x

    sent1_train, sent2_train, sent1_test, sent2_test = [], [], [], []
    label_train, label_dev, label_test = [], [], []

    if _train:
        with open(trainloc, 'r', encoding='utf8') as f:
            f.readline()  # skipping the header of the file
            for line in f:
                text = line.strip().split('\t')
                sent1_train.append(_preprocess_sentence(text[3]))
                sent2_train.append(_preprocess_sentence(text[4]))
                label_train.append(int(text[0]))

    if _test:
        with open(testloc, 'r', encoding='utf8') as f:
            f.readline()  # skipping the header of the file
            for line in f:
                text = line.strip().split('\t')
                sent1_test.append(_preprocess_sentence(text[3]))
                sent2_test.append(_preprocess_sentence(text[4]))
                label_test.append(int(text[0]))

    if _train and _test:
        return [sent1_train, sent2_train], [sent1_test, sent2_test], [label_train, label_test]
    elif _train:
        return [sent1_train, sent2_train], label_train
    elif _test:
        return [sent1_test, sent2_test], label_test


class DataGenerator:
    @classmethod
    def get_train_data(cls):
        [sent1_train, sent2_train], label_train = load_data(_preprocess_sentence=None, _train=True, _test=False)
        return [
            {"s1": item[0], "s2": item[1], "label": item[2]}
            for item in zip(sent1_train, sent2_train, label_train)
        ]

    @classmethod
    def get_test_data(cls):
        [sent1_test, sent2_test], label_test = load_data(_preprocess_sentence=None, _train=False, _test=True)

        return [
            {"s1": item[0], "s2": item[1], "label": item[2]}
            for item in zip(sent1_test, sent2_test, label_test)
        ]


idf_model = TfIdf(DataGenerator.get_test_data())


def get_spacy_module():
    return spacy.load('en_core_web_sm')
    # return en_core_web_md.load()


nlp = get_spacy_module()


def get_dependancy_graph(s, display=False):
    doc = nlp(s)
    if display:
        spacy.displacy.render(doc, style="dep", jupyter=True)
    edges = []
    nodes = [{
        "node": "ROOT",
        "token": None,
        "is_fake": True,
    }]
    for token in doc:
        nodes.append({
            "node": token.text,
            "token": token,
            "is_fake": False,
        })
        if token.dep_ == "ROOT":
            edges.append({
                "start": "ROOT",
                "end": token.text,
                "start_node_id": 0,
                "end_node_id": token.i + 1,
                "type": token.dep_
            })
        else:
            edges.append({
                "start": token.head.text,
                "end": token.text,
                "start_node_id": token.head.i + 1,
                "end_node_id": token.i + 1,
                "type": token.dep_
            })
    return {"nodes": nodes, "edges": edges}


class HungarianGraphNodesMatcher:

    def __init__(self, _g1, _g2, threshold=0.5):
        self.g1 = _g1
        self.g2 = _g2
        self.node_threshold = threshold
        self.create_cost_matrix()
        self.solve_linear_sum_assignment()
        self.match_nodes()

    def set_threshold(self, threshold):
        self.node_threshold = threshold
        self.match_nodes()

    def create_cost_matrix(self):
        self.matrix = np.zeros((len(self.g1["nodes"]), len(self.g2["nodes"])))
        for i1, n1 in enumerate(self.g1["nodes"]):
            for i2, n2 in enumerate(self.g2["nodes"]):
                if (not n1["is_fake"] and not n2["is_fake"] and
                        n1["token"].has_vector and n2["token"].has_vector):
                    self.matrix[i1][i2] = n1["token"].similarity(n2["token"])
                elif n1["is_fake"] == n2["is_fake"]:
                    self.matrix[i1][i2] = n1["node"] == n2["node"]
                else:
                    self.matrix[i1][i2] = 0

        # Now we need to fleep scores, because Hungarian is trying to minimize
        self.cost = np.subtract(np.full(self.matrix.shape, 1), self.matrix)

    def get_pandas_matrix(self):
        df = pd.DataFrame(
            data=self.matrix,
            index=np.array([n["node"] for n in self.g1["nodes"]]),
            columns=np.array([n["node"] for n in self.g2["nodes"]])
        )

        return df

    def solve_linear_sum_assignment(self):
        row_ind, col_ind = linear_sum_assignment(self.cost)

        self.row_ind = row_ind
        self.col_ind = col_ind

    def match_nodes(self):
        self.graph1_to_graph2 = {
            item[0]: item[1]
            for item in zip(self.row_ind, self.col_ind)
            if self.matrix[item[0]][item[1]] > self.node_threshold
        }

    def create_node_aliases(self):
        for id1, n1 in enumerate(self.g1["nodes"]):
            n1["alias"] = "G1_" + str(id1) + n1["node"]
        for id2, n2 in enumerate(self.g2["nodes"]):
            n2["alias"] = "G2_" + str(id2) + n2["node"]
        for id1, id2 in self.graph1_to_graph2.items():
            n1 = self.g1["nodes"][id1]
            n2 = self.g2["nodes"][id2]
            n1["alias"] = "G1_" + str(id1) + "_" + n1["node"] + "_G2_" + str(id2) + "_" + n2["node"]
            n2["alias"] = n1["alias"]

    def build_graph(self, g):
        nx_g = nx.Graph()
        for edge in g["edges"]:
            start_node = g["nodes"][edge["start_node_id"]]
            end_node = g["nodes"][edge["end_node_id"]]
            nx_g.add_edge(start_node["alias"], end_node["alias"])
        return nx_g

    def get_converted_graphs(self):
        self.create_node_aliases()
        g1 = self.build_graph(self.g1)
        g2 = self.build_graph(self.g2)
        return g1, g2

    def print_matched_nodes(self):
        print ("Graph 1  =>   Graph 2")
        for id1, id2 in self.graph1_to_graph2.items():
            print(f"{self.g1['nodes'][id1]['node']}    =>   {self.g2['nodes'][id2]['node']}")


class HungarianGraphFeatureGenerator:
    NAME = 'HungarianGraph'

    def get_features_for_graphs(self, node_matcher, similarity):
        node_matcher.set_threshold(similarity)
        g1, g2 = node_matcher.get_converted_graphs()
        score_normalized = compare_graphs(g1, g2, False, True)
        score_raw = compare_graphs(g1, g2, False, False)
        return np.array([score_normalized, score_raw])

    def get_features(self, s1, s2):
        g1 = get_dependancy_graph(s1, False)
        g2 = get_dependancy_graph(s2, False)
        node_matcher = HungarianGraphNodesMatcher(g1, g2, 0.9)

        features = np.array([])

        for similarity in [0.8, 0.85, 0.90, 0.95]:
            features = np.append(features, self.get_features_for_graphs(node_matcher, similarity))

        return features


class HungarianNodeFeatureGenerator:
    NAME = 'HungarianNode'

    def get_features_for_graphs(self, node_matcher, similarity):
        node_matcher.set_threshold(similarity)
        g1, g2 = node_matcher.get_converted_graphs()
        n1, n2 = len(g1), len(g2)
        num_matched_nodes = len(node_matcher.graph1_to_graph2)
        percent_matched = num_matched_nodes * 2. / (n1 + n2)
        features = np.array([n1, n2, percent_matched])
        return features

    def get_features(self, s1, s2):
        g1 = get_dependancy_graph(s1, False)
        g2 = get_dependancy_graph(s2, False)
        node_matcher = HungarianGraphNodesMatcher(g1, g2, 0.9)

        features = np.array([])

        for similarity in [0.8, 0.85, 0.90, 0.95]:
            features = np.append(features, self.get_features_for_graphs(node_matcher, similarity))

        return features


class PathFeatureGenerator:
    NAME = 'PathSimilarity'

    SIMILARITY = 0.8

    def get_feature_for_length(self, g_f1, g_f2, length):
        f1 = g_f1.get_path_features(length=length)
        f2 = g_f2.get_path_features(length=length)

        norm = len(f1) + len(f2)

        features = np.array([])
        for similarity in [0.8, 0.85, 0.90, 0.95]:
            score = MatchFeatureVectors.match_feature_vectors(f1, f2, similarity)
            score = (score * 2.) / norm if norm != 0 else 0

            features = np.append(features, score)

        return features

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        features = np.array([])
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 0))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 1))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 2))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 3))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 4))

        return features


class SubtreeFeatureGenerator:
    NAME = 'SubtreeFeature'

    SIMILARITY = 0.8

    def get_feature_for_length(self, g_f1, g_f2, length):
        f1 = g_f1.get_subtree_features(length=length)
        f2 = g_f2.get_subtree_features(length=length)

        norm = len(f1) + len(f2)

        features = np.array([])
        for similarity in [0.8, 0.85, 0.90, 0.95]:
            score = MatchFeatureVectors.match_feature_vectors(f1, f2, similarity)
            score = (score * 2.) / norm if norm != 0 else 0

            features = np.append(features, score)

        return features

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        features = np.array([])
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 0))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 1))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 2))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 3))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 4))

        return features


class RootNodeFeatureGenerator:
    NAME = 'RootNodeFeature'

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        root_node1 = GraphBuilder.get_root_node(g1)
        root_node2 = GraphBuilder.get_root_node(g2)

        if root_node1['token'].has_vector and root_node2['token'].has_vector:
            score = root_node1['token'].similarity(root_node2['token'])
        else:
            score = 0

        features = np.array([
            score,
        ])

        return features


class SimpleEdgeMatcher:
    NAME = 'SimpleEdgeMatcher'

    SIMILARITY = 0.8

    def simple_match_edges(self, g_f1, g_f2):
        f1 = g_f1.get_simple_edge_features()
        f2 = g_f2.get_simple_edge_features()

        score = 0
        for edge1 in f1:
            if (edge1['start_node']['token'] is None or
                    not edge1['start_node']['token'].has_vector
                    or edge1['end_node']['token'] is None
                    or not edge1['end_node']['token'].has_vector):
                continue
            for edge2 in f2:
                if (edge2['start_node']['token'] is None
                        or not edge2['start_node']['token'].has_vector
                        or edge2['end_node']['token'] is None
                        or not edge2['end_node']['token'].has_vector):
                    continue
                if (Vector.similarity(
                        edge1['start_node']['token'].vector,
                        edge2['start_node']['token'].vector
                ) > self.SIMILARITY
                        and Vector.similarity(
                            edge1['end_node']['token'].vector,
                            edge2['end_node']['token'].vector
                        ) > self.SIMILARITY
                ):
                    score += 1

        similarity_score = (1. * score) / (len(f1) * len(f2))

        return similarity_score

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        features = np.array([
            self.simple_match_edges(g_f1, g_f2)
        ])

        return features


class SimpleEdgeMatcherWithDependancy:
    NAME = 'SimpleEdgeMatcherWithDependancy'

    SIMILARITY = 0.8

    def simple_match_edges_with_dependancy_type(self, g_f1, g_f2):
        f1 = g_f1.get_simple_edge_features()
        f2 = g_f2.get_simple_edge_features()

        score = 0
        total = 0
        for edge1 in f1:
            if (edge1['start_node']['token'] is None or
                    not edge1['start_node']['token'].has_vector
                    or edge1['end_node']['token'] is None
                    or not edge1['end_node']['token'].has_vector):
                continue
            for edge2 in f2:
                if (edge2['start_node']['token'] is None
                        or not edge2['start_node']['token'].has_vector
                        or edge2['end_node']['token'] is None
                        or not edge2['end_node']['token'].has_vector):
                    continue
                if (Vector.similarity(
                        edge1['start_node']['token'].vector,
                        edge2['start_node']['token'].vector
                ) > self.SIMILARITY
                        and Vector.similarity(
                            edge1['end_node']['token'].vector,
                            edge2['end_node']['token'].vector
                        ) > self.SIMILARITY
                ):
                    if (edge1['dependancy_type'] == edge2['dependancy_type']):
                        score += 1
                    total += 1

        similarity_score = 0 if total == 0 else (1. * score) / total

        return similarity_score

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        features = np.array([
            self.simple_match_edges_with_dependancy_type(g_f1, g_f2)
        ])

        simple_edge_matcher_feature_generator = SimpleEdgeMatcher()
        features = np.append(features, simple_edge_matcher_feature_generator.get_features(s1, s2))

        return features


class SimpleApproximateBigramKernel:
    """
    There was an error here while training!, probably better to remove this feature.
          From https://www.aclweb.org/anthology/L16-1452.pdf
          Simple Approximate Bigram Kernel (SABK)
    """

    NAME = 'SimpleApproximateBigramKernel'
    EDGE_SIMILARITY_SCORE = 2

    @classmethod
    def node_similarity(cls, node1, node2):
        if (node1['token'] is None or
                node2['token'] is None or
                not node1['token'].has_vector or
                not node2['token'].has_vector
        ):
            return 1 if node1['node'] == node2['node'] else 0
        else:
            return Vector.similarity(
                node1['token'].vector,
                node2['token'].vector
            )

    @classmethod
    def edge_similarity(cls, edge1, edge2):
        return SimpleApproximateBigramKernel.EDGE_SIMILARITY_SCORE if edge1['dependancy_type'] == edge2[
            'dependancy_type'] else 1

    @classmethod
    def similarity(cls, edge1, edge2):
        start_node_similarity = cls.node_similarity(edge1['start_node'], edge2['start_node'])
        end_node_similarity = cls.node_similarity(edge1['end_node'], edge2['end_node'])

        edge_similarity = cls.edge_similarity(edge1, edge2)

        return (start_node_similarity + end_node_similarity) * edge_similarity

    @classmethod
    def compute_simple_approximate_bigram_kernel(cls, g_f1, g_f2):
        f1 = g_f1.get_simple_edge_features()
        f2 = g_f2.get_simple_edge_features()

        similarity_score = 0

        for edge1 in f1:
            for edge2 in f2:
                similarity_score += cls.similarity(edge1, edge2)

        similarity_score = (similarity_score * 1.) / (len(g_f1.g.nodes) + len(g_f2.g.nodes))

        return similarity_score

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        features = np.array([
            SimpleApproximateBigramKernel.compute_simple_approximate_bigram_kernel(g_f1, g_f2)
        ])

        return features


class SubtreeFeatureGeneratorIdf:
    NAME = 'SubtreeFeatureIdf'

    SIMILARITY = 0.8

    def get_feature_for_length(self, g_f1, g_f2, length):
        f1 = g_f1.get_subtree_features(length=length, idf_model=idf_model)
        f2 = g_f2.get_subtree_features(length=length, idf_model=idf_model)

        norm = len(f1) + len(f2)

        features = np.array([])
        for similarity in [0.8, 0.85, 0.90, 0.95]:
            score = MatchFeatureVectors.match_feature_vectors(f1, f2, similarity)
            score = (score * 2.) / norm if norm != 0 else 0

            features = np.append(features, score)

        return features

    def get_features(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        features = np.array([])
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 0))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 1))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 2))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 3))
        features = np.append(features, self.get_feature_for_length(g_f1, g_f2, 4))

        return features


class MarchFeatureGenerator:
    NAME = 'MarchFeature'

    def get_feature_1(self, s1, s2):
        len_s1 = GeneralFeatures.get_s_len(s1)
        len_s2 = GeneralFeatures.get_s_len(s2)

        def f(len_s1, len_s2):
            d_1 = (len_s1 - len_s2) * 1. / len_s1
            d_2 = 1. / 0.8 ** (len_s1 - len_s2)
            r = np.array([d_1, d_2])
            return r

        feature_1 = np.array([])
        feature_1 = np.append(feature_1, f(len_s1, len_s2))
        feature_1 = np.append(feature_1, f(len_s2, len_s1))
        return feature_1

    def get_feature_2(self, s1, s2):
        doc_1 = nlp(s1)
        doc_2 = nlp(s2)

        def compare_n_grams(s1, s2, doc_1, doc_2, n):
            s1_list = GeneralFeatures.get_n_grams(s1, n, doc_1)
            s2_list = GeneralFeatures.get_n_grams(s2, n, doc_2)

            def is_n_gram_equal(n_gram_1, n_gram_2):
                for i in range(len(n_gram_1)):
                    if n_gram_1[i].text != n_gram_2[i].text:
                        if n_gram_1[i].similarity(n_gram_2[i]) < 0.9:
                            return False
                return True

            count = 0
            for n_gram_1 in s1_list:
                match = False
                for n_gram_2 in s2_list:
                    if is_n_gram_equal(n_gram_1, n_gram_2):
                        match = True
                if match:
                    count += 1
            d = count * 1. / len(s1_list) if len(s1_list) > 0 else 0
            return np.array([d])

        feature_2 = np.array([])

        feature_2 = np.append(feature_2, compare_n_grams(s1, s2, doc_1, doc_2, 1))
        feature_2 = np.append(feature_2, compare_n_grams(s2, s1, doc_2, doc_1, 1))
        feature_2 = np.append(feature_2, compare_n_grams(s1, s2, doc_1, doc_2, 2))
        feature_2 = np.append(feature_2, compare_n_grams(s2, s1, doc_2, doc_1, 2))
        feature_2 = np.append(feature_2, compare_n_grams(s1, s2, doc_1, doc_2, 3))
        feature_2 = np.append(feature_2, compare_n_grams(s2, s1, doc_2, doc_1, 3))

        return feature_2

    def get_feature_4(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        g_f1 = GraphFeatures(g1)
        g_f2 = GraphFeatures(g2)

        f1 = g_f1.get_simple_edge_features()
        f2 = g_f2.get_simple_edge_features()

        def edge_similarity(edge1, edge2):
            return (
                    (edge1['dependancy_type'] == edge2['dependancy_type'])
                    and NodeSimilarity.basic(edge1['start_node'], edge2['start_node']) > 0.9
                    and NodeSimilarity.basic(edge1['end_node'], edge2['end_node']) > 0.9
            )

        def get_dependancy_similarity(f1, f2):
            similarity_score = 0

            for edge1 in f1:
                match = False
                for edge2 in f2:
                    if edge_similarity(edge1, edge2):
                        match = True
                if match:
                    similarity_score += 1

            similarity_score = (similarity_score * 1.) / len(f1) if len(f1) > 0 else 0

            return np.array([similarity_score])

        feature_4 = np.array([])
        feature_4 = np.append(feature_4, get_dependancy_similarity(f1, f2))
        feature_4 = np.append(feature_4, get_dependancy_similarity(f2, f1))

        return feature_4

    def get_feature_5(self, s1, s2):
        g1 = GraphBuilder.build_nx_graph_from_sentance(s1)
        g2 = GraphBuilder.build_nx_graph_from_sentance(s2)

        def compare_n_grams(g1, g2, length):
            # Length in traversal starts with 0
            length = length - 1
            traversal_1 = GraphTraversal(graph=g1)
            traversal_2 = GraphTraversal(graph=g2)

            s1_list = traversal_1.get_all_paths_with_len(length=length)
            s2_list = traversal_2.get_all_paths_with_len(length=length)

            def is_n_gram_equal(g1, g2, n_gram_1, n_gram_2):
                for i in range(len(n_gram_1)):
                    if NodeSimilarity.basic(g1.nodes[n_gram_1[i]], g2.nodes[n_gram_2[i]]) < 0.9:
                        return False
                return True

            count = 0
            for n_gram_1 in s1_list:
                match = False
                for n_gram_2 in s2_list:
                    if is_n_gram_equal(g1, g2, n_gram_1, n_gram_2):
                        match = True
                if match:
                    count += 1
            d = count * 1. / len(s1_list) if len(s1_list) > 0 else 0
            return np.array([d])

        feature_5 = np.array([])

        feature_5 = np.append(feature_5, compare_n_grams(g1, g2, 1))
        feature_5 = np.append(feature_5, compare_n_grams(g2, g1, 1))
        feature_5 = np.append(feature_5, compare_n_grams(g1, g2, 2))
        feature_5 = np.append(feature_5, compare_n_grams(g2, g1, 2))
        feature_5 = np.append(feature_5, compare_n_grams(g1, g2, 3))
        feature_5 = np.append(feature_5, compare_n_grams(g2, g1, 3))
        feature_5 = np.append(feature_5, compare_n_grams(g1, g2, 4))
        feature_5 = np.append(feature_5, compare_n_grams(g2, g1, 4))

        return feature_5

    def get_feature_6(self, s1, s2):

        def get_bleu(s1, s2, n_grams):
            return np.array([BLEUCalculator.compute(
                s1,
                s2,
                GeneralFeatures.get_n_grams,
                NGramSimilarity.basic_word,
                n_grams
            )])

        feature_6 = np.array([])

        feature_6 = np.append(feature_6, get_bleu(s1, s2, 1))
        feature_6 = np.append(feature_6, get_bleu(s2, s1, 1))
        feature_6 = np.append(feature_6, get_bleu(s1, s2, 2))
        feature_6 = np.append(feature_6, get_bleu(s2, s1, 2))
        feature_6 = np.append(feature_6, get_bleu(s1, s2, 3))
        feature_6 = np.append(feature_6, get_bleu(s2, s1, 3))
        feature_6 = np.append(feature_6, get_bleu(s1, s2, 4))
        feature_6 = np.append(feature_6, get_bleu(s2, s1, 4))

        return feature_6

    def get_features(self, s1, s2):
        features = np.array([])

        features = np.append(features, self.get_feature_6(s1, s2))

        return features


class MarchFeatureGeneratorWithoutBleu(MarchFeatureGenerator):
    NAME = 'MarchFeatureGeneratorWithoutBleu'

    def get_features(self, s1, s2):
        features = np.array([])

        features = np.append(features, self.get_feature_1(s1, s2))
        features = np.append(features, self.get_feature_2(s1, s2))
        features = np.append(features, self.get_feature_4(s1, s2))
        features = np.append(features, self.get_feature_5(s1, s2))

        return features


class MarchFeatureGeneratorOnlyBleu(MarchFeatureGenerator):
    NAME = 'MarchFeatureGeneratorOnlyBleu'

    def get_features(self, s1, s2):
        features = np.array([])

        features = np.append(features, self.get_feature_6(s1, s2))

        return features


class AllFeatureFinal:
    NAME = 'AllFeatureFinal'

    def get_features(self, s1, s2):
        generators = [
            HungarianGraphFeatureGenerator(),
            HungarianNodeFeatureGenerator(),
            PathFeatureGenerator(),
            SubtreeFeatureGenerator(),
            RootNodeFeatureGenerator(),
            SimpleEdgeMatcher(),
            SimpleEdgeMatcherWithDependancy(),
            SimpleApproximateBigramKernel(),
            SubtreeFeatureGeneratorIdf(),
            MarchFeatureGeneratorWithoutBleu(),
            MarchFeatureGeneratorOnlyBleu()
        ]
        features = np.array([])
        for generator in generators:
            features = np.append(features, generator.get_features(s1, s2))
        return features


# Code is taken from https://github.com/Jacobe2169/ged4py

class EdgeGraph():
    def __init__(self, init_node, nodes):
        self.init_node = init_node
        self.nodes_ = nodes

    def nodes(self):
        return self.nodes_

    def size(self):
        return len(self.nodes)

    def __len__(self):
        return len(self.nodes_)


class AbstractGraphEditDistance(object):
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

    def normalized_distance(self):
        """
        Returns the graph edit distance between graph g1 & g2
        The distance is normalized on the size of the two graphs.
        This is done to avoid favorisation towards smaller graphs
        """
        avg_graphlen = len(self.g1) + len(self.g2)
        return self.distance() / avg_graphlen

    def distance(self):
        return sum(self.edit_costs())

    def edit_costs(self):
        cost_matrix = self.create_cost_matrix()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return [cost_matrix[row_ind[i]][col_ind[i]] for i in range(len(row_ind))]

    def create_cost_matrix(self):
        """
        Creates a |N+M| X |N+M| cost matrix between all nodes in
        graphs g1 and g2
        Each cost represents the cost of substituting,
        deleting or inserting a node
        The cost matrix consists of four regions:
        substitute 	| insert costs
        -------------------------------
        delete 		| delete -> delete
        The delete -> delete region is filled with zeros
        """
        n = len(self.g1)
        m = len(self.g2)
        cost_matrix = np.zeros((n + m, n + m))
        # cost_matrix = [[0 for i in range(n + m)] for j in range(n + m)]
        nodes1 = self.g1.nodes() if float(nxv) < 2 else list(self.g1.nodes())
        nodes2 = self.g2.nodes() if float(nxv) < 2 else list(self.g2.nodes())

        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = self.substitute_cost(nodes1[i], nodes2[j])

        for i in range(m):
            for j in range(m):
                cost_matrix[i + n, j] = self.insert_cost(i, j, nodes2)

        for i in range(n):
            for j in range(n):
                cost_matrix[j, i + m] = self.delete_cost(i, j, nodes1)

        self.cost_matrix = cost_matrix
        return cost_matrix

    def insert_cost(self, i, j):
        raise NotImplementedError

    def delete_cost(self, i, j):
        raise NotImplementedError

    def substitute_cost(self, nodes1, nodes2):
        raise NotImplementedError

    def print_matrix(self):
        print("cost matrix:")
        for column in self.create_cost_matrix():
            for row in column:
                if row == sys.maxsize:
                    print ("inf\t")
                else:
                    print ("%.2f\t" % float(row))
            print("")


class EdgeEditDistance(AbstractGraphEditDistance):
    """
    Calculates the graph edit distance between two edges.
    A node in this context is interpreted as a graph,
    and edges are interpreted as nodes.
    """

    def __init__(self, g1, g2):
        AbstractGraphEditDistance.__init__(self, g1, g2)

    def insert_cost(self, i, j, nodes2):
        if i == j:
            return 1
        return sys.maxsize

    def delete_cost(self, i, j, nodes1):
        if i == j:
            return 1
        return sys.maxsize

    def substitute_cost(self, edge1, edge2):
        if edge1 == edge2:
            return 0.
        return 1


class GraphEditDistance(AbstractGraphEditDistance):
    def __init__(self, g1, g2):
        AbstractGraphEditDistance.__init__(self, g1, g2)

    def substitute_cost(self, node1, node2):
        return self.relabel_cost(node1, node2) + self.edge_diff(node1, node2)

    def relabel_cost(self, node1, node2):
        if node1 == node2:
            return 0.
        else:
            return 1.

    def delete_cost(self, i, j, nodes1):
        if i == j:
            return 1
        return sys.maxsize

    def insert_cost(self, i, j, nodes2):
        if i == j:
            return 1
        else:
            return sys.maxsize

    def pos_insdel_weight(self, node):
        return 1

    def edge_diff(self, node1, node2):
        edges1 = list(self.g1.edge[node1].keys()) if float(nxv) < 2 else list(self.g1.edges(node1))
        edges2 = list(self.g2.edge[node2].keys()) if float(nxv) < 2 else list(self.g2.edges(node2))
        if len(edges1) == 0 or len(edges2) == 0:
            return max(len(edges1), len(edges2))

        edit_edit_dist = EdgeEditDistance(EdgeGraph(node1, edges1), EdgeGraph(node2, edges2))
        return edit_edit_dist.normalized_distance()


def compare_graphs(g1, g2, print_details=False, use_normalized=True):
    ged = GraphEditDistance(g1, g2)

    if print_details:
        ged.print_matrix()

    return ged.normalized_distance() if use_normalized else ged.distance()


class GraphBuilder:

    def __init__(self):
        pass

    @classmethod
    def build_nx_graph_from_dt(cls, g):
        nx_g = nx.Graph()
        for index, node in enumerate(g["nodes"]):
            nx_g.add_node(index, node=node['node'], token=node['token'], is_fake=node['is_fake'])
        for edge in g["edges"]:
            nx_g.add_edge(edge["start_node_id"], edge["end_node_id"], dependancy_type=edge["type"])
        return nx_g

    @classmethod
    def build_nx_graph_from_sentance(cls, s):
        graph = get_dependancy_graph(s, False)
        return cls.build_nx_graph_from_dt(graph)

    @classmethod
    def get_root_node(cls, g):
        main_root_node = [n for n, _ in g.adj[0].items()][0]
        return g.nodes[main_root_node]


class GraphFeatures:
    def __init__(self, graph=None, sentance=None):
        assert graph is not None or sentance is not None
        if graph is not None:
            self.g = graph
        else:
            self.g = GraphBuilder.build_nx_graph_from_sentance(sentance)

    def get_path_features(self, length=0):
        traversal = GraphTraversal(graph=self.g)
        pathes = traversal.get_all_paths_with_len(length=length)

        pathes_with_nodes = [
            [self.g.nodes[node] for node in path]
            for path in pathes
        ]

        filtered_pathes_with_nodes = [
            path
            for path in pathes_with_nodes
            if all(
                node["token"] is not None and node["token"].has_vector
                for node in path
            )
        ]

        aggregated_vectors = [
            sum([node["token"].vector for node in path])
            for path in filtered_pathes_with_nodes
        ]

        return aggregated_vectors

    def get_subtree_features(self, length=0, remove_tree_without_vector=True, remove_stop_words=False, idf_model=None):
        """
        Return list of vectors, where each vector represent one subtree.
        Subtree is created by aggregating vectors in this subtree.

        Keyword arguments:
        length -- the real part (default 0.0)
        remove_tree_without_vector -- remove whole tree if at least one vector inside it
          is empty (non common word)
        remove_stop_words - remove word from tree if it is stop word
        idf_model - If present, multiply vector by word idf
        """
        traversal = GraphTraversal(graph=self.g)
        subtrees = traversal.get_all_subtrees_with_depth(length=length)

        subtrees_with_nodes = [
            [self.g.nodes[node] for node in subtree]
            for subtree in subtrees
        ]

        if remove_tree_without_vector:
            subtrees_with_nodes = [
                subtree
                for subtree in subtrees_with_nodes
                if all(
                    node["token"] is not None and node["token"].has_vector
                    for node in subtree
                )
            ]

        idf = lambda word: 1

        if idf_model is not None:
            idf = lambda word: idf_model.get_idf(word)

        aggregated_vectors = [
            sum([
                node["token"].vector * idf(node["token"])
                for node in subtree
                # If remove_tree_without_vector == false
                if node["token"] is not None and node["token"].has_vector
                and (not remove_stop_words or not node["token"].is_stop)
            ])
            for subtree in subtrees_with_nodes
        ]

        # Filter empty vectors
        aggregated_vectors = [
            v
            for v in aggregated_vectors
            if not np.isscalar(v)
        ]

        return aggregated_vectors

    def get_simple_edge_features(self):
        """
        Return list of edges
        """
        edges = []
        for (start_idx, end_idx, dependancy_type) in self.g.edges.data('dependancy_type'):
            item = {}
            item['start_idx'] = start_idx
            item['end_idx'] = end_idx
            item['dependancy_type'] = dependancy_type
            item['start_node'] = self.g.nodes[start_idx]
            item['end_node'] = self.g.nodes[end_idx]
            edges.append(item)

        return edges


class GraphTraversal:
    def __init__(self, graph=None, sentance=None):
        assert graph is not None or sentance is not None
        if graph is not None:
            self.g = graph
        else:
            self.g = GraphBuilder.build_nx_graph_from_sentance(sentance)

    def get_paths_from_root_to_leafs(self, root=0):
        #                 node. parent. path.
        res, stack = [], [(root, None, [])]
        while stack:
            node, parent, path = stack.pop()
            path.append(node)
            neighbours = [n for n, _ in self.g.adj[node].items()]
            if len(neighbours) == 1 and neighbours[0] == parent:
                res.append(path)
            for n in neighbours:
                if n == parent:
                    continue
                stack.append((n, node, path[:]))
        return res

    def get_all_paths_with_len(self, root=0, length=0):
        """
        Return list of pathes with specificified len + 1.
        The start is every node.

        For the tree:
               1
             2   3
           5
             6

        Len = 2:
        [1, 2, 5]
        [2, 5, 6]

        """

        #                  node. parent. path.
        res, stack = [], [(root, None, [])]
        started_new_path = {root}
        while stack:
            node, parent, path = stack.pop()
            path.append(node)
            neighbours = [n for n, _ in self.g.adj[node].items()]
            if len(path) == length + 1:
                res.append(path)
            for n in neighbours:
                if n == parent:
                    continue
                if len(path) < length + 1:
                    stack.append((n, node, path[:]))
                if n not in started_new_path:
                    stack.append((n, node, []))
                    started_new_path.add(n)

        return res

    def get_all_subtrees_with_depth(self, root=0, parent=None, length=0):
        """
          Return array of subtrees.
          Each subtree is defined by indexes of their nodes.
        """
        #                  node. parent. distance.
        res, stack = [], [(root, parent, 0)]
        reached_depth = False
        while stack:
            node, _parent, distance = stack.pop()
            res.append(node)
            if distance >= length:
                reached_depth = True
                continue
            neighbours = [n for n, _ in self.g.adj[node].items()]
            for n in neighbours:
                if n == _parent:
                    continue
                stack.append((n, node, distance + 1))

        all_subtrees = []
        if reached_depth:
            all_subtrees.append(res)
        for n, _ in self.g.adj[root].items():
            if n == parent:
                continue
            all_subtrees += self.get_all_subtrees_with_depth(n, root, length)

        return all_subtrees


class Vector:
    @classmethod
    def get_norm(cls, v):
        total = (v ** 2).sum()
        return np.sqrt(total) if total != 0 else 0

    @classmethod
    def similarity(cls, v1, v2):
        v1_norm = cls.get_norm(v1)
        v2_norm = cls.get_norm(v2)
        if v1_norm == 0 or v1_norm == 0:
            return 0.0
        return (np.dot(v1, v2) / (v1_norm * v2_norm))


class MatchFeatureVectors:
    @classmethod
    def match_feature_vectors(cls, features1, features2, similarity=0.8):
        """
          This function tries to do the following:
          1) For each vector in features1 try to find whether vector with good similarity exist in features2.
          Return ammount of matched vectors.
        """
        count = 0

        features1_norm = []
        for v1 in features1:
            for v2 in features2:
                score = Vector.similarity(v1, v2)
                if score > similarity:
                    count += 1
                    break
        return count


class GeneralFeatures:

    @classmethod
    def get_s_len(cls, s):
        doc = nlp(s)
        return np.array([len(doc)])

    @classmethod
    def get_n_grams(cls, s, n, doc=None):
        if doc is None:
            d = nlp(s)
        else:
            d = doc

        res = []
        count = 0
        for token in d[:len(d) - n + 1]:
            res.append(d[count:count + n])
            count = count + 1
        return res


class NodeSimilarity:

    @classmethod
    def basic(cls, node1, node2):
        if (node1['token'] is None or
                node2['token'] is None or
                not node1['token'].has_vector or
                not node2['token'].has_vector
        ):
            return 1 if node1['node'] == node2['node'] else 0
        else:
            return Vector.similarity(
                node1['token'].vector,
                node2['token'].vector
            )

    @classmethod
    def token_similarity(cls, token1, token2):
        """
        It's spacy tokens
        """
        if token1.has_vector and token2.has_vector:
            return Vector.similarity(
                token1.vector,
                token2.vector
            )
        else:
            return 1 if token1.text == token2.text else 0


class BrevityPenalty:

    @classmethod
    def compute(cls, ref_length, hyp_length):
        """
            ref_lengths - int
            hyp_lengths - int
            Return BrevityPenalty - double
            https://www.nltk.org/_modules/nltk/translate/bleu_score.html
        """

        if hyp_length > ref_length:
            return 1
        # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
        elif hyp_length == 0:
            return 0
        else:
            return math.exp(1 - ref_length / hyp_length)


class BLEUCalculator:

    @classmethod
    def precision(cls, reference, hypothesis, get_n_grams_funct, is_n_gram_equal_func, n):
        """
        s1 - First sentance
        s2 - Second sentance
        get_n_grams_funct - Function that takes:
            s - sentance
            n - size of n gram
            Return list of n_grams
        is_n_gram_equal_func - n_gram comparator
            n_gram_1
            n_gram_2
            Return Bool
        n - size of bigram

        """
        # Extracts all ngrams in hypothesis
        # Set an empty Counter if hypothesis is empty.

        reference_n_grams = get_n_grams_funct(reference, n)

        hypothesis_n_grams = get_n_grams_funct(hypothesis, n)

        total_found = 0

        for n_gram_h in hypothesis_n_grams:
            found = False
            for n_gram_r in reference_n_grams:
                #                 print(n_gram_h)
                #                 print(n_gram_r)
                if is_n_gram_equal_func(n_gram_h, n_gram_r):
                    #                     print(n_gram_h)
                    #                     print(n_gram_r)
                    #                     print("Match")
                    #                     print("*" * 20)

                    found = True
            if found:
                total_found += 1

        numerator = total_found
        # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
        denominator = max(1, len(hypothesis_n_grams))

        return (numerator * 1.) / denominator

    @classmethod
    def compute(cls, reference, hypothesis, get_n_grams_funct, is_n_gram_equal_func, max_n):
        """
        s1 - First sentance
        s2 - Second sentance
        get_n_grams_funct - Function that takes:
            s - sentance
            n - size of n gram
            Return list of n_grams
        is_n_gram_equal_func - n_gram comparator
            n_gram_1
            n_gram_2
            Return Bool
        max_n - Max size of bigram

        Return BLEU - double

        https://www.nltk.org/_modules/nltk/translate/bleu_score.html
        """

        p_n = []

        weight = 1. / max_n

        weights = [weight] * max_n

        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            _p = cls.precision(reference, hypothesis, get_n_grams_funct, is_n_gram_equal_func, i)
            if abs(_p) < 0.001:
                return 0
            p_n.append(_p)

        hyp_lengths = GeneralFeatures.get_s_len(hypothesis)
        ref_lengths = GeneralFeatures.get_s_len(reference)

        # Calculate brevity penalty.
        bp = BrevityPenalty.compute(ref_lengths, hyp_lengths)

        s = [w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n)]
        s = bp * math.exp(math.fsum(s))

        return s

    @classmethod
    def test_precision(cls):
        """
            Example from https://leimao.github.io/blog/BLEU-Score/

        """
        s1 = "the cat is on the mat"
        s2 = "the cat the cat on the mat"

        p1 = cls.precision(
            s1,
            s2,
            GeneralFeatures.get_n_grams,
            NGramSimilarity.basic_word,
            1
        )

        assert (abs(p1 - 1.) < 0.001)

        p2 = cls.precision(
            s1,
            s2,
            GeneralFeatures.get_n_grams,
            NGramSimilarity.basic_word,
            2
        )

        assert (abs(p2 - 0.66666) < 0.001)

    @classmethod
    def test_bleu(cls):
        """
            Example from https://leimao.github.io/blog/BLEU-Score/

        """
        s1 = "the cat is on the mat"
        s2 = "the cat the cat on the mat"

        bleu1 = cls.compute(
            s1,
            s2,
            GeneralFeatures.get_n_grams,
            NGramSimilarity.basic_word,
            4
        )
        print(bleu1)


class NGramSimilarity:

    @classmethod
    def basic_word(cls, n_gram_1, n_gram_2):
        """
            n_gram_1 is Token
            n_gram_2 is Token
        """
        if isinstance(n_gram_1[0], SpacyToken):
            comparator = NodeSimilarity.token_similarity

        for i in range(len(n_gram_1)):
            if comparator(n_gram_1[i], n_gram_2[i]) < 0.9:
                return False
        return True


def features_for_prediction(s1, s2):
    feature_generator = AllFeatureFinal()
    features = feature_generator.get_features(s1, s2)

    bitmask = [False, True, True, False, False, False, False, False, False, True, True, True, True, False, True, True,
               False, True, True, True, True, True, False, False, True, False, False, False, False, True, True, True,
               True, True, False, True, True, True, True, True, True, True, True, True, False, True, False, True, False,
               True, True, False, False, False, True, False, True, False, False, True, True, False, False, True, True,
               True, False, True, True, True, False, True, False, False, False, False, True, False, True, True, True,
               True, True, False, True, True, False, True, True, False, False, True, False, False, False, False, True,
               True, True, False, True, False, False, False, True, True, True, True, True, False, False, True, False]

    features_bm = features[bitmask].reshape(1, -1)

    return features_bm
