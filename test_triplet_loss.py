"""Test for the triplet loss computation."""

import unittest

import numpy as np
import tensorflow as tf

from model.model_fn import compute_triplet_loss


def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
            pairwise_distances.diagonal())
    return pairwise_distances


class TripletLossTest(unittest.TestCase):
    """Basic test cases."""

    def test_triplet_loss(self):
        num_img_per_class = 4
        num_classes = 3
        num_data = num_img_per_class * num_classes
        feat_dim = 6
        margin = 1.0

        embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
        labels = np.zeros(num_data)
        for label in range(num_classes):
            start = label * num_img_per_class
            end = start + num_img_per_class
            labels[start:end] = label

        pdist_matrix = pairwise_distance_np(embeddings, squared=True)

        loss_np = 0.0
        num_positives = 0.0
        num_valid = 0.0
        for i in range(num_data):
            for j in range(num_data):
                for k in range(num_data):
                    distinct = (i != j and i != j and j != k)
                    valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                    if distinct and valid:
                        num_valid += 1.0

                        pos_distance = pdist_matrix[i][j]
                        neg_distance = pdist_matrix[i][k]

                        loss = np.maximum(0.0, pos_distance - neg_distance + margin)
                        loss_np += loss

                        if loss > 0:
                            num_positives += 1.0

        loss_np /= num_positives

        # Compute the loss in TF.
        loss_tf, fraction = compute_triplet_loss(embeddings, num_classes, num_img_per_class, margin)
        with tf.Session() as sess:
            loss_tf_val, fraction_val = sess.run([loss_tf, fraction])
        assert np.allclose(loss_np, loss_tf_val)
        assert np.allclose(num_positives / num_valid, fraction_val)



if __name__ == '__main__':
    unittest.main()

