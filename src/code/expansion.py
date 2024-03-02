from collections import Counter

import numpy as np


def calculate_term_probabilities(query, feedback_docs):
    query_term_counts = Counter(query)
    feedback_term_counts = Counter(feedback_docs)

    query_length = len(query)
    feedback_length = len(feedback_docs)

    query_probabilities = {term: count / query_length for term, count in query_term_counts.items()}
    feedback_probabilities = {term: count / feedback_length for term, count in feedback_term_counts.items()}

    return query_probabilities, feedback_probabilities


def calculate_kl_divergence(query_probabilities, feedback_probabilities):
    kl_divergence = {}

    for term, feedback_probability in feedback_probabilities.items():
        query_probability = query_probabilities.get(term, 0.001)  # Avoid division by zero
        kl_divergence[term] = feedback_probability * np.log(feedback_probability / query_probability)

    return kl_divergence


def expand_query(query, feedback_docs, num_terms_to_add):
    query_probabilities, feedback_probabilities = calculate_term_probabilities(query, feedback_docs)
    kl_divergence = calculate_kl_divergence(query_probabilities, feedback_probabilities)

    # Select the terms with the highest KL divergence
    new_terms = sorted(kl_divergence, key=kl_divergence.get, reverse=True)[:num_terms_to_add]

    # Add the new terms to the query
    expanded_query = query + new_terms

    return expanded_query
