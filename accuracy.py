import json

"""
accuracy uses the average overlap approach to create a simple, yet powerful way
to measure the performance of our model. In our case we only need to know the
output file and construct the "ground truth" on our own. We assume that for
every document there exist 10 similar documents that we will want to have a
similarity of 1. This approach is a weighted (higher ranks have more impact) and
non-conjoint (items can vary between the two lists) measure.
Original Paper: http://codalism.com/research/papers/wmz10_tois.pdf
"""

# _similarity_score penalizes wrong predictions with high similarity and
# rewards correct predictions with high similarity
def _similarity_score(pred, sim):
    return 0.5 + sim/2 if pred == 1 else (1-sim)/2

# _document_score is the ratio of correct / wrong predictions and also factors
# in the similarity score at rank t
def _document_score(correct_label, sim_t, num_equal, num_total):
    return 0.5*(num_equal / num_total) + 0.5*sim_t

# _average_overlap is the mean of
def _average_overlap(a_ts, num_total):
    return sum(a_ts) / num_total

def accuracy(filepath):
    data = json.load(open(filepath, 'r'))
    overlap_scores = []
    for k, v in data.items():
        correct_label = v['label']
        num_equal = 0
        num_total = 0
        ao_t = 0
        a_ts = []
        for doc in v['most_similars']:
            num_total += 1
            pred = 1 if doc['label'] == correct_label else 0
            num_equal += pred
            sim_t = _similarity_score(pred, doc['similarity'])
            a_t = _document_score(correct_label, sim_t, num_equal, num_total)
            a_ts.append(a_t)
        ao_t = _average_overlap(a_ts, num_total)
        overlap_scores.append(ao_t)
    return sum(overlap_scores) / len(overlap_scores)
