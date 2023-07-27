import numpy as np
import scipy.sparse as sp
import utils
from hyperparameters import ALPHA, EPSILON, TRACKING_METHOD, LOAD_INSTEAD_OF_RECALCULATION, LOAD_MAX, DATASET_NAME, ABLATION
if DATASET_NAME == 'movielens-1m':
    from dataprocessing_temporal_movielens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
elif DATASET_NAME == 'bitcoinalpha':
    from dataprocessing_temporal_bitcoinalpha import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
elif DATASET_NAME == 'wikilens':
    from dataprocessing_temporal_wikilens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
import time
import pdb
HAVE_NODE_ATTR = 1
HAVE_EDGE_ATTR = 1

def get_graph_info(graph_file_name, node_attr_file_name):
    node_attr_set = set()
    edge_attr_set = set()
    graph_file = open(graph_file_name)
    node_attr_file = open(node_attr_file_name)
    for line in graph_file.readlines():
        items = line.strip().split(',')
        edge_attr = int(items[2])
        edge_attr_set.add(edge_attr)
    for line in node_attr_file.readlines():
        items = line.strip().split(',')
        node_attr = int(items[1])
        node_attr_set.add(node_attr)
    graph_file.close()
    node_attr_file.close()
    node_attr_list = []
    edge_attr_list = []
    for node_attr in node_attr_set:
        node_attr_list.append(node_attr)
    for edge_attr in edge_attr_set:
        edge_attr_list.append(edge_attr)
    return len(node_attr_set), len(edge_attr_set), node_attr_list, edge_attr_list


def build_matrices_accelerated(graph_file_name, node_attr_file_name, num_node_attr):
    graph_file = graph_file_name
    nodes_set = set()
    node_attr_file_open = open(node_attr_file_name, 'r')
    # first going through to count number of nodes
    for line in node_attr_file_open.readlines():
        items = line.strip().split(',')
        node = int(items[0])
        nodes_set.add(node)
    nnodes = len(nodes_set)
    adj_matrix = sp.lil_matrix((nnodes, nnodes))
    node_attr_matrix = sp.lil_matrix((nnodes, nnodes))
    node_attr_matrix_acce = sp.lil_matrix((nnodes, num_node_attr))
    edge_attr_matrix = sp.lil_matrix((nnodes, nnodes)) 
    node_attr_file_open.close()

    # second going through to set up the adjacency (and possibly edge attribute) matrix
    graph_file_open = open(graph_file, 'r')
    for line in graph_file_open.readlines():
        items = line.strip().split(',')
        node_0 = int(items[0])
        node_1 = int(items[1])
        adj_matrix[node_0, node_1] = 1
        adj_matrix[node_1, node_0] = 1
        if (HAVE_EDGE_ATTR != 0):
            edge_attr_matrix[node_0, node_1] = int(items[2])
            edge_attr_matrix[node_1, node_0] = int(items[2])
    graph_file_open.close()

    # set up the node attribute matrix using a seperated file
    if (HAVE_NODE_ATTR != 0):
        node_attr_file_open = open(node_attr_file_name, 'r')
        for line in node_attr_file_open.readlines():
            items = line.strip().split(',')
            node = int(items[0])
            attr_of_node = int(items[1])
            node_attr_matrix_acce[node, attr_of_node] = 1
            node_attr_matrix[node, node] = attr_of_node

    return nnodes, adj_matrix, node_attr_matrix, edge_attr_matrix, node_attr_matrix_acce


if __name__ == '__main__':
    start_time = time.time()
    num_node_attr, num_edge_attr, node_attr_list, edge_attr_list = get_graph_info('datasets/' + DATASET_NAME + '-temporal/edge.txt', 'datasets/' + DATASET_NAME + '-temporal/node_attr.txt')
    nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, node_attr_matrix_acce_1 = build_matrices_accelerated(
        'datasets/' + DATASET_NAME + '-temporal/edge_sorted_initial_sub.txt', 'datasets/' + DATASET_NAME + '-temporal/node_attr_sub.txt', max(node_attr_list) + 1
        )
    nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, node_attr_matrix_acce_2 = build_matrices_accelerated(
        'datasets/' + DATASET_NAME + '-temporal/edge_sorted_initial.txt', 'datasets/' + DATASET_NAME + '-temporal/node_attr.txt', max(node_attr_list) + 1
        )
    initial_graph_file = open('datasets/' + DATASET_NAME + '-temporal/edge_sorted_initial.txt', 'r')
    start_line = len(initial_graph_file.readlines())
    initial_graph_file.close()
    graph_file = open('datasets/' + DATASET_NAME + '-temporal/edge_sorted.txt', 'r')
    all_line = graph_file.readlines()
    graph_file.close()
    end_line = len(all_line)

    # data structure for pagerank vector updating
    P_dict = {}
    M = {}
    q = {}
    v = {}
    time_record = []

    # data structure for filtering
    all_nodes_in_whole = set()
    for node in range(nnodes_2):
        all_nodes_in_whole.add(node)
    candidate = {node: all_nodes_in_whole.copy() for node in range(nnodes_1)}
    F = {}  # F((j, i)) will be used to record which (NodeAttribute, EdgeAttribute) filtered node j in G_2 away from candidate[i]

    # get the ground truth match
    ground_truth_file = open('datasets/' + DATASET_NAME + '-temporal/ground_truth.txt', 'r')
    ground_truth = {}
    for line in ground_truth_file.readlines():
        items = line.strip().split(',')
        node_in_whole = int(items[0])
        node_in_sub = int(items[1])
        ground_truth[node_in_sub] = node_in_whole
    

    # pre-computing
    if not LOAD_INSTEAD_OF_RECALCULATION:
        # transition matrix P
        P = utils.calculate_transition_matrix(node_attr_list, edge_attr_list, nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, node_attr_matrix_acce_1, nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, node_attr_matrix_acce_2)
        # pre-knowledge h
        h = utils.calculate_pre_knowledge_h(nnodes_1, nnodes_2, candidate, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, F)

        sp.save_npz('saved_data_' + DATASET_NAME + '/P_matrix_start.npz', P)
        np.save('saved_data_' + DATASET_NAME + '/h_array_start.npy', h)
    else:
        P = sp.load_npz('saved_data_' + DATASET_NAME + '/P_matrix_start.npz')
        h = np.load('saved_data_' + DATASET_NAME + '/h_array_start.npy')
    
    P_dict[0] = P
    h_uniform = utils.uniform_h(nnodes_1, nnodes_2)

    # the one-hot solution
    if (TRACKING_METHOD == 3):
        P = P.tocsr()
        number = 0
        onehot_ppr = utils.calc_onehot_ppr_matrix(P, ALPHA, 2).tolil().transpose()
        for i in range(nnodes_1 * nnodes_2):
            q[i] = onehot_ppr[i, :]
            M[i] = P_dict[0]
            if (number % 100000) == 0:
                print(number)
            number += 1
    
    pre_end_time = time.time()
    print("data processing and precomputing time:", pre_end_time - start_time)
    # initial pagerank vector
    if (ABLATION):
        print('setting h_0 to uniform distribution in ablation study')
        h = h_uniform.copy()
    v[0] = utils.calc_ppr_by_power_iteration(P, ALPHA, h, 20)
    exp_match = utils.greedy_match(v[0], nnodes_1, nnodes_2)
    hit_rate1 = utils.check_greedy_hit1_with_return(exp_match, ground_truth, nnodes_1)
    acc_record = []
    acc_record.append(hit_rate1)

    t = 0
    counter = 0
    for line_index in range(start_line, end_line):
        if (counter == LOAD_MAX):
            pdb.set_trace()
        sub_change = 0
        items = all_line[line_index].strip().split(',')
        node1 = int(items[0])
        node2 = int(items[1])
        edge_attr = int(items[2])
        t += 1
        adj_matrix_2[node1, node2] = 1
        adj_matrix_2[node2, node1] = 1
        edge_attr_matrix_2[node1, node2] = edge_attr
        edge_attr_matrix_2[node2, node1] = edge_attr

        if is_in_subgraph(node1) and is_in_subgraph(node2):
            if edge_attr_matrix_1[match_from_whole_to_sub(node1), match_from_whole_to_sub(node2)] != edge_attr:
                sub_change = 1
        if (sub_change == 0):
            continue
        # if comes to here, sub graph is changed
        counter += 1
        print("time:", t)
        print("line in all edges:", line_index + 1)
        print("subchange:", counter)
        node_sub_1 = match_from_whole_to_sub(node1)
        node_sub_2 = match_from_whole_to_sub(node2)
        adj_matrix_1[node_sub_1, node_sub_2] = 1
        adj_matrix_1[node_sub_2, node_sub_1] = 1
        edge_attr_matrix_1[node_sub_1, node_sub_2] = edge_attr
        edge_attr_matrix_1[node_sub_2, node_sub_1] = edge_attr

        if LOAD_INSTEAD_OF_RECALCULATION and (counter <= LOAD_MAX):
            P_new = sp.load_npz('./saved_data_' + DATASET_NAME + '/P_matrix_' + str(counter) + '.npz')
            h_new_sp = sp.load_npz('./saved_data_' + DATASET_NAME + '/h_array_sp_' + str(counter) + '.npz')
            h_new = h_new_sp.toarray().ravel()

        else:
            P_new = utils.calculate_transition_matrix(node_attr_list, edge_attr_list, nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, node_attr_matrix_acce_1, nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, node_attr_matrix_acce_2)
            h_new = utils.calculate_pre_knowledge_h(nnodes_1, nnodes_2, candidate, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, F)
            sp.save_npz('./saved_data_' + DATASET_NAME + '/P_matrix_' + str(counter) + '.npz', P_new)
            h_new_sp = sp.csr_matrix(h_new)
            sp.save_npz('./saved_data_' + DATASET_NAME + '/h_array_sp_' + str(counter) + '.npz', h_new_sp)

        time1 = time.time()

        if TRACKING_METHOD == 2:
            v[counter] = utils.tracking_method_two(v[counter-1], P, P_new, h_new, ALPHA, EPSILON)
        else: # TRACKING_METHOD == 3
            P_dict[counter] = P_new.copy()
            v_mid = utils.osp(v[counter-1], P, P_new, ALPHA, EPSILON, 1)
            delta_h = h_new - h
            print("number of changed elements in h:", np.count_nonzero(delta_h))
            for i in range(nnodes_1 * nnodes_2):
                if delta_h[i] != 0:
                    q_new = utils.osp(q[i].toarray().ravel(), M[i], P_new, ALPHA, EPSILON, 0)
                    q[i] = sp.lil_matrix(q_new)
                    M[i] = P_dict[counter]
                    v_mid = v_mid + delta_h[i] * q_new
            v[counter] = v_mid

        time2 = time.time()
        print("tracking time: ", time2 - time1)
        time_record.append(time2 - time1)
        P = P_new.copy()
        h = h_new.copy()

        exp_match = utils.greedy_match(v[counter], nnodes_1, nnodes_2)
        hit_rate1 = utils.check_greedy_hit1_with_return(exp_match, ground_truth, nnodes_1)
        acc_record.append(hit_rate1)

    pdb.set_trace()