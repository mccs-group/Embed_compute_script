import numpy as np
import heapq
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time


def cfg_init_matrices(size: int, adj_list):
    adjacency_matrix = np.zeros((size, size))
    D_0_diag = np.zeros(size)
    for i in range(0, len(adj_list), 2):
        pred_id = adj_list[i]
        succ_id = adj_list[i + 1]
        adjacency_matrix[pred_id, succ_id] = 1
        D_0_diag[pred_id] += 1

    D_0_inv = np.zeros((size, size))
    for i in range(size):
        D_0_inv[i, i] = 0 if D_0_diag[i] == 0 else 1 / D_0_diag[i]

    mu = [1 if np.nonzero(D_0_row)[0].size == 0 else 0 for D_0_row in D_0_inv]

    return adjacency_matrix, D_0_inv, mu


def get_P(adj_mat, D_0_inv, mu, petr_factor: float, size: int):
    ones = np.ones((size, 1))
    mu = np.transpose([mu])
    P = (1 - petr_factor) * (
        np.matmul(D_0_inv, adj_mat) + np.matmul(mu, np.transpose(ones)) * (1.0 / size)
    )
    P += petr_factor * (1 / size) * np.ones((size, size))

    return P


def get_Phi_inverse_L(P):
    val, vec = linalg.eig(np.transpose(P))
    pi = []

    for i, eigen_val in enumerate(val):
        if abs(eigen_val.real - 1) < 1e-6 and (abs(eigen_val.imag) < 1e-6):
            pi = vec[:, i].real
            break

    pi = pi / np.sum(pi)

    Phi = np.diagflat(pi)
    Phi_inv = linalg.inv(Phi)
    identity_mat = np.identity(P.shape[0])
    result = identity_mat - 0.5 * (
        P + np.matmul(np.matmul(Phi_inv, np.transpose(P)), Phi)
    )
    return result


def get_embed_vec(Phi_inv_L, K: int):
    embed_val, embed_vec = linalg.eig(Phi_inv_L)

    # print("eigen vals:")
    # print(embed_val)

    # print("eigen vecs:")
    # print(embed_vec)
    # print("ended printing")

    sorted_vals = sorted(embed_val)

    heap_sorted = np.unique(heapq.nsmallest(K + 1, embed_val)[1:])
    # print("sorted")
    # print(heap_sorted)

    final_embed = np.zeros((Phi_inv_L.shape[0], K))
    embed_vec = embed_vec.real

    i = 0
    while i < K:
        for index in np.where(np.isclose(embed_val, sorted_vals[i]))[0]:
            final_embed[:, i] = embed_vec[:, index].flatten()
            i += 1

    # print("got vectors: ")
    # print(final_embed)

    return final_embed


def compress_pca(embedding, dims=1, random_state=25):
    pca_obj = PCA(n_components=dims, random_state=random_state)

    return pca_obj.fit_transform(np.transpose(embedding))


def get_microsoft_cfg_embed(adj_list, K: int, petr_factor: float):
    start = time.time()

    size = adj_list[0]
    if (K + 1) > adj_list[0]:
        size = K + 1

    adjacency_matrix, D_0_inv, mu = cfg_init_matrices(size, adj_list[1:])
    P = get_P(adjacency_matrix, D_0_inv, mu, 0.01, size)
    Phi_inv_L = get_Phi_inverse_L(P)
    vec = get_embed_vec(Phi_inv_L, K)

    end = time.time()
    print(f"delta: {end - start}")

    return compress_pca(vec)


# for val flow


def get_val_flow_mat(size: int, adj_list):
    adjacency_matrix = np.zeros((size, size))
    for i in range(0, len(adj_list), 2):
        adjacency_matrix[adj_list[i], adj_list[i + 1]] = 1

    return adjacency_matrix


def get_proximity_mat(adj_matrix, beta=0.8, H=3):
    adj_mat_accum = np.identity(adj_matrix.shape[0])
    beta_accum = 1
    M = np.zeros((adj_matrix.shape[0], adj_matrix.shape[0]))
    for i in range(H):
        adj_mat_accum = np.matmul(adj_mat_accum, adj_matrix)
        M += adj_mat_accum * beta_accum
        beta_accum *= beta
    return M


def get_svd_vec(proximity_mat, K: int):
    U, S, V_h = linalg.svd(proximity_mat)
    V = np.transpose(V_h)
    D_src = U[:, 0:K]
    D_dst = V[:, 0:K]

    return D_src, D_dst


def get_flow2vec_embed(
    adj_list, K: int, beta=0.8, H=3, ndims=1, compress_random_state=25
):
    size = max(K, adj_list[0])
    def_use_matrix = get_val_flow_mat(size, adj_list[1:])
    prox_mat = get_proximity_mat(def_use_matrix, beta, H)
    D_src_emb, D_dst_emb = get_svd_vec(prox_mat, K)
    D_src_emb = np.where(abs(D_src_emb) < 1e-10, 0, D_src_emb)
    D_dst_emb = np.where(abs(D_dst_emb) < 1e-10, 0, D_dst_emb)
    # print("got from svd:")
    # print(list(D_src_emb))
    # print("and")
    # print(list(D_dst_emb))
    D_src_compressed = compress_pca(D_src_emb, ndims, compress_random_state)
    D_dst_compressed = compress_pca(D_dst_emb, ndims, compress_random_state)
    # print(D_src_compressed)
    # print(D_dst_compressed)
    return np.concatenate((D_src_compressed, D_dst_compressed), axis=None)
