import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
import time
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import random

import nltk
nltk.download('punkt', quiet=True)

np.random.seed(42)
random.seed(42)

class VectorOptimizer:
    def __init__(self):
        self.best_params = None
        self.best_score = np.inf

    def _tokenize(self, data, text_column):
        data = data.copy()
        data['tokens'] = data[text_column].apply(lambda x: word_tokenize(str(x).lower()))
        return data

    @staticmethod
    def _get_word2vec_features(tokens, model):
        features = []
        for token in tokens:
            if token in model.wv:
                features.append(model.wv[token])
        return sum(features) / len(features) if features else []

    @staticmethod
    def _evaluate_vectorization(vectors):
        similarities = cosine_similarity(vectors)
        return np.mean(similarities)

    def _analyze_vector(self, sampled_data, vectorizer_type, text_column, params=None, verbose=0):
        try:
            if vectorizer_type == 'tf-idf':
                vectorizer = TfidfVectorizer(**params) if params else TfidfVectorizer()
                term_matrix = vectorizer.fit_transform(sampled_data[text_column])
                params = vectorizer.get_params()
                score = self._evaluate_vectorization(term_matrix.toarray())

            elif vectorizer_type == 'bow':
                vectorizer = CountVectorizer(**params) if params else CountVectorizer()
                term_matrix = vectorizer.fit_transform(sampled_data[text_column])
                params = vectorizer.get_params()
                score = self._evaluate_vectorization(term_matrix.toarray())

            elif vectorizer_type == 'word2vec':  # word2vec
                sampled_data = self._tokenize(sampled_data, text_column)
                if not params:
                    params = {
                        'vector_size': 100, 'window': 5, 'min_count': 5,
                        'workers': 1, 'sg': 0, 'hs': 0, 'epochs': 5
                    }
                model = Word2Vec(sampled_data['tokens'], **params)
                sampled_data['word2vec_features'] = sampled_data['tokens'].apply(self._get_word2vec_features, args=(model,))
                term_matrix = pd.DataFrame(sampled_data['word2vec_features'].tolist())
                term_matrix = term_matrix.fillna(0)
                score = self._evaluate_vectorization(term_matrix.to_numpy())

            else:
                raise ValueError("Invalid option, choose 'tf-idf', 'bow', or 'word2vec'.")

        except Exception as e:
            if str(e) == "Invalid option, choose 'tf-idf', 'bow', or 'word2vec'.":
                raise ValueError("Invalid option, choose 'tf-idf', 'bow', or 'word2vec'.")
            if verbose > 0:
                print(f"Error analyzing vectorization: {e}")
            score = np.inf
        return params, score
                

    def optimize_vectorizer(self, data, text_column, vectorizer_type='tf-idf', parameters_grid=None, verbose=0):
        self.best_params = None
        self.best_score = np.inf

        if parameters_grid is None:
            parameters_grid = self._get_default_parameters_grid(vectorizer_type)

        total_combos = len(ParameterGrid(parameters_grid))
        start_time = time.time()

        if total_combos == 1:
            params = next(ParameterGrid(parameters_grid))
            if verbose > 0:
                print(f"\n[1/1] Using parameters: {params}")
            self.best_params, self.best_score = self._analyze_vector(data, vectorizer_type, text_column, params, verbose=verbose)

        else:
            if verbose > 0:
                print('Optimizing vectorizer hyperparameters...')
                print("\nInitializing with default parameters...")
            self.best_params, self.best_score = self._analyze_vector(data, vectorizer_type, text_column, verbose=verbose)
            if verbose > 0:
                print(f"Default parameters: {self.best_params}")
                print(f"Initial similarity: {self.best_score}")

            for i, params in enumerate(ParameterGrid(parameters_grid), 1):
                if verbose > 0:
                    print(f"\n[{i}/{total_combos}] Testing parameters: {params}")
                current_params, current_score = self._analyze_vector(data, vectorizer_type, text_column, params, verbose=verbose)

                if current_score < self.best_score:
                    self.best_params, self.best_score = current_params, current_score

                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (total_combos - i)
                if verbose > 0:
                    print(f"Similarity: {current_score} | Best: {self.best_score}")
                    print(f"Time: {elapsed:.2f}s | Remaining: {remaining:.2f}s")

        if verbose > 0:
            print("\nBest parameters:", self.best_params)
            print("Best similarity:", self.best_score)
        return self.best_params, self.best_score

    def _get_default_parameters_grid(self, vectorizer_type):
        if vectorizer_type == 'tf-idf':
            return {
                'max_features': [100, 1000, 5000],
                'ngram_range': [(1, 1), (1, 2), (1, 3)],
                'min_df': [1, 10],
                'max_df': [0.8, 1.0],
                'norm': ['l1', 'l2']
            }
        elif vectorizer_type == 'bow':
            return {
                'max_features': [100, 1000, 5000],
                'ngram_range': [(1, 1), (1, 2), (1, 3)],
                'min_df': [1, 10],
                'max_df': [0.8, 1.0],
                'binary': [True, False]
            }
        else:
            return {
                'vector_size': [100, 1000, 5000],
                'window': [3, 5, 7],
                'min_count': [5, 10],
                'sg': [0, 1],
                'hs': [0, 1],
                'workers': [1]
            }