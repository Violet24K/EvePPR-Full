import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import math

def calc_ppr_by_power_iteration(P: sp.spmatrix, alpha: float, h: np.ndarray, t: int) -> np.ndarray:
    iterated = (1 - alpha) * h
    result = iterated.copy()
    for iteration in range(t):
        iterated = (alpha * P).dot(iterated)
        result += iterated
    # last_norm = np.linalg.norm(iterated, 1)
    # print("last_norm:", last_norm)
    return result


def calc_onehot_ppr_matrix(P: sp.spmatrix, alpha: float, t: int) -> np.ndarray:
    iterated = (1 - alpha) * sp.eye(P.shape[0])
    matrix_result = iterated.copy()
    for iteration in range(t):
        iterated = (alpha * P).dot(iterated)
        matrix_result += iterated
    return matrix_result            


# udpate v from $\mathbf{v} =  \alpha \mathbf{A} \mathbf{v} + (1-\alpha) \mathbf{h}$ to $\mathbf{v} =  \alpha \mathbf{B} \mathbf{v} + (1-\alpha) \mathbf{h}$
def osp(v: np.ndarray, A: sp.spmatrix, B: sp.spmatrix, alpha: float, epsilon: float, whether_print: int) -> np.ndarray:
    assert A.shape == B.shape, "in osp, the dimension of matrix A should be the same as the dimension of matrix B"
    q_offset = alpha * (B - A) @ v
    v_offset = q_offset.copy()
    x_offset = q_offset.copy()
    number = 0
    # pdb.set_trace()
    while (np.linalg.norm(x_offset, 1) > epsilon):
        number += 1 
        x_offset = alpha * B @ x_offset
        v_offset += x_offset
    if whether_print == 1:
        # print("x_offset norm of osp:", np.linalg.norm(x_offset, 1), "iteration number of osp:", number)
        pass
    return v + v_offset


# udpate v from a previous solution. The return value approximately satisfies $\mathbf{v} =  \alpha \mathbf{P} \mathbf{v} + (1-\alpha) \mathbf{h}$
def gauss_southwell(v: np.ndarray, P: sp.spmatrix, h: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    dimension_P = P.shape[0]
    x = v
    r = (1 - alpha) * h - (sp.eye(dimension_P) - alpha * P) @ v
    max_index = np.argmax(r)
    number = 0
    while r[max_index] > epsilon:
        e = np.zeros(dimension_P,)
        e[max_index] = 1
        x = x + r[max_index] * e
        r = r - r[max_index] * e + alpha * r[max_index] * P @ e
        max_index = np.argmax(r)
        number += 1 
    # print("final residual maximum element:", r[np.argmax(r)])
    return x


# udpate v from $\mathbf{v} =  \alpha \mathbf{P_old} \mathbf{v} + (1-\alpha) \mathbf{h_old}$ to $\mathbf{v} =  \alpha \mathbf{P_new} \mathbf{v} + (1-\alpha) \mathbf{h_new}$
def tracking_method_two(v: np.ndarray, P_old: sp.spmatrix, P_new: sp.spmatrix, h_new:np.ndarray, alpha: float, epsilon: float):
    v_mid = osp(v, P_old, P_new, alpha, epsilon, 1)
    return gauss_southwell(v_mid, P_new, h_new, alpha, epsilon)


def neighbor(a: int, adj_matrix: sp.lil_matrix):
    neighbor_set = set()
    for column_index in adj_matrix.rows[a]:
        neighbor_set.add(column_index)
    return neighbor_set


def node_attribute_filter(x, a, candidate, node_attr_matrix_1, node_attr_matrix_2):
    if x in candidate[a]:
        if (node_attr_matrix_1[a, a] != node_attr_matrix_2[x, x]):
            candidate[a].remove(x)


def one_hop_filter(x, a, candidate, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2):
    invested_pair = set()
    if x not in candidate[a]:
        return None
    for b in neighbor(a, adj_matrix_1):
        exist = 0
        node_edge_attr_pair = (node_attr_matrix_1[b, b], edge_attr_matrix_1[a, b])
        if node_edge_attr_pair in invested_pair:
            continue
        for y in neighbor(x, adj_matrix_2):
            if ( node_edge_attr_pair == (node_attr_matrix_2[y, y], edge_attr_matrix_2[x, y])):
                exist = 1
                break
        invested_pair.add(node_edge_attr_pair)
        if exist == 0:
            candidate[a].remove(x)
            return node_edge_attr_pair
    return None


def construct_pre_knowledge(candidate, nnodes_1, nnodes_2):
    h = np.zeros(nnodes_1 * nnodes_2)
    for i in range(nnodes_1):
        if len(candidate[i]) != 0:
            weight = 1/(nnodes_1 * len(candidate[i]))
            index = i * nnodes_2
            for j in range(nnodes_2):
                if j in candidate[i]:
                    h[index + j] = weight
    return h


def calculate_pre_knowledge_h(nnodes_1, nnodes_2, candidate, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, F):
    for node_in_1 in range(nnodes_1):
        for node_in_2 in range(nnodes_2):
            node_attribute_filter(node_in_2, node_in_1, candidate, node_attr_matrix_1, node_attr_matrix_2)
            filtering_pair = one_hop_filter(node_in_2, node_in_1, candidate, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2)
            if filtering_pair != None:
                F[(node_in_2, node_in_1)] = filtering_pair
    h = construct_pre_knowledge(candidate, nnodes_1, nnodes_2)
    print("h constructed")
    return h


def uniform_h(nnodes_1, nnodes_2):
    h = np.zeros(nnodes_1 * nnodes_2)
    value = 1/(nnodes_1 * nnodes_2)
    for i in range(nnodes_1):
        for j in range(nnodes_2):
            h[i * nnodes_2 + j] = value
    return h


def greedy_match(s: np.ndarray, nnodes_1, nnodes_2):
    min_size = min(nnodes_1, nnodes_2)
    used_rows = np.zeros(nnodes_2)
    used_cols = np.zeros(nnodes_1)
    exp_match = np.zeros(min_size)
    row = np.zeros(min_size)
    col = np.zeros(min_size)
    y = -np.sort(-s)
    ix = np.argsort(-s)
    matched = 0
    index = 0

    while(matched < min_size):
        ipos = ix[index]
        jc = math.floor(ipos/nnodes_2)
        ic = ipos-jc*nnodes_2
        if (used_rows[ic] != 1) and (used_cols[jc] != 1):
            row[matched] = ic
            col[matched] = jc
            exp_match[jc] = ic
            used_rows[ic] = 1
            used_cols[jc] = 1
            matched += 1
        index += 1
    return exp_match


# imputs are from sub to whole
def check_greedy_hit1(exp_match, actual_match, n_sub_node):
    hit = 0
    for i in range(n_sub_node):
        if (exp_match[i] == actual_match[i]):
            hit += 1
    print('greedy hit rate1:', hit/n_sub_node)


def check_greedy_hit1_with_return(exp_match, actual_match, n_sub_node):
    hit = 0
    for i in range(n_sub_node):
        if (exp_match[i] == actual_match[i]):
            hit += 1
    result = hit/n_sub_node
    print('greedy hit rate1:', result)
    return result



def calculate_transition_matrix(node_attr_list, edge_attr_list, nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, node_attr_matrix_acce_1, nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, node_attr_matrix_acce_2):
    N_1_K = {}
    N_2_K = {}
    E_1_L = {}
    E_2_L = {}
    for node_attr_index in range(len(node_attr_list)):
        node_attr = node_attr_list[node_attr_index]
        N_1_k_matrix = sp.lil_matrix((nnodes_1, nnodes_1))
        for i in range(nnodes_1):
            if (node_attr_matrix_1[i, i] == node_attr):
                N_1_k_matrix[i, i] = 1
        N_1_K[node_attr_index] = N_1_k_matrix
        N_2_k_matrix = sp.lil_matrix((nnodes_2, nnodes_2))
        for i in range(nnodes_2):
            if (node_attr_matrix_2[i, i] == node_attr):
                N_2_k_matrix[i, i] = 1
        N_2_K[node_attr_index] = N_2_k_matrix
    for edge_attr_index in range(len(edge_attr_list)):
        edge_attr = edge_attr_list[edge_attr_index]
        E_1_l_matrix = sp.lil_matrix((nnodes_1, nnodes_1))
        for row_index in range(len(adj_matrix_1.rows)):
            for column_index in adj_matrix_1.rows[row_index]:
                if (edge_attr_matrix_1[row_index, column_index] == edge_attr):
                    E_1_l_matrix[row_index, column_index] = 1
        E_1_L[edge_attr_index] = E_1_l_matrix
        E_2_l_matrix = sp.lil_matrix((nnodes_2, nnodes_2))
        for row_index in range(len(adj_matrix_2.rows)):
            for column_index in adj_matrix_2.rows[row_index]:
                if (edge_attr_matrix_2[row_index, column_index] == edge_attr):
                    E_2_l_matrix[row_index, column_index] = 1
        E_2_L[edge_attr_index] = E_2_l_matrix
    
    # construct N and E matrix
    print("constructing N matrix")
    N = sp.csr_matrix((nnodes_1*nnodes_2, nnodes_1*nnodes_2))
    for node_attr_index in range(len(node_attr_list)):
        to_add = sp.kron(N_1_K[node_attr_index], N_2_K[node_attr_index], 'csr')
        N = N + to_add

    print("constructing E matrix")
    E = sp.csr_matrix((nnodes_1*nnodes_2, nnodes_1*nnodes_2))
    for edge_attr_index in range(len(edge_attr_list)):
        to_add = sp.kron(E_1_L[edge_attr_index], E_2_L[edge_attr_index], 'csr')
        E = E + to_add

    # construct the W (nnodes_1*nnodes_2, nnodes_1*nnodes_2) matrix
    print("constructing W matrix")
    W = (N.dot(E.multiply(sp.kron(adj_matrix_1, adj_matrix_2)))).dot(N)
    
    # construct the D matrix for normalization
    print("constructing D matrix")
    d = sp.csr_matrix((nnodes_1*nnodes_2, 1))
    for k in range(len(node_attr_list)):
        for l in range(len(edge_attr_list)):
            first_term = (E_1_L[l].multiply(adj_matrix_1)).dot(node_attr_matrix_acce_1[:, node_attr_list[k]])
            second_term = (E_2_L[l].multiply(adj_matrix_2)).dot(node_attr_matrix_acce_2[:, node_attr_list[k]])
            d += sp.kron(first_term, second_term, "csr")

    dd = np.zeros(nnodes_1 * nnodes_2)
    rcv = sp.find(d)
    for line in range(len(rcv[0])):
        if rcv[2][line] != 0:
            dd[rcv[0][line]] = (rcv[2][line])**(-1/2)

    D = sp.lil_matrix((nnodes_1 * nnodes_2, nnodes_1 * nnodes_2))
    for i in range(nnodes_1 * nnodes_2):
        if dd[i] != 0:
            D[i, i] = dd[i]

    # the symmetric normalized matrix, also the transition matrix that will be used in the network alignment problem
    print("constructing W_symmetric_normalized matrix")
    W_symmetric_normalized = (D.dot(W)).dot(D)
    print("transition matrix constructed")
    return W_symmetric_normalized