import numpy as np
from hyper_graph import HyperGraph

class FormulaReader:

    APPLICATION   = 0 #"*"
    ABSTRACTION   = 1 #"/"
    FREE_VAR      = 2 #"f"
    BOUNDED_VAR   = 3 #"b"
    UNKNOWN_CONST = 4 #不明な語
    FIRST_CONST   = 5

    def __init__(self, ver2 = False):

        self.edge_arities = (3,3)
        self.input_mul = sum(x*(x-1) for x in self.edge_arities)+1
        self.ver2 = ver2

    def set_vocab(self, reverse_vocabulary_index, vocabulary_index):

        self.vocabulary = []
        self.vocab_size = self.FIRST_CONST
        for word in vocabulary_index:
            if word == '*': self.vocabulary.append(self.APPLICATION)
            elif word == '/': self.vocabulary.append(self.ABSTRACTION)
            elif word[0] == 'f': self.vocabulary.append(self.FREE_VAR)
            elif word[0] == 'b': self.vocabulary.append(self.BOUNDED_VAR)
            #*,/,f,bの時,self.vocab_sizeの値をself.vocabularyリストに追加.
            else:
                self.vocabulary.append(self.vocab_size)
                self.vocab_size += 1
        print("vocab_size: {}".format(self.vocab_size))
        print("vocablaryを数値割当→extract_subformulaで使用（論理式の抽出の時の条件分岐）")
        print("self.vocabulary:{}".format(self.vocabulary))

    def __call__(self, formula_list, preselection):

        #preselection is None
        if preselection is not None:
            #事前選択はグラフリストではまだサポートされていません．
            raise Exception("preselection is not supported for GraphLists yet")

        #GraphListインスタンス呼び出し
        #第一引数：(3,3),第２引数:GraphConv2Layer
        #print("GraphListインスタンス呼び出し")
        graph_list = GraphList(self.edge_arities, ver2 = self.ver2)

        #formula_list = [tokens]:textをインデックス番号に変換したもの
        for formula in formula_list:
            self.index = 0
            self.variables = dict()

            #HyperGraphインスタンス生成
            #print("HyperGraphインスタンス呼び出し")
            self.graph = HyperGraph()

            #token(textをリストのインデックスに変換をしたもの)
            self.line = formula

            #print("extract_subformula呼び出し：論理式の抽出,グラフ構築")
            #print("root=頂点")
            root = self.extract_subformula()#subformula抽出

            #assert文:条件式がTrueでない時に，例外を投げる
            #条件式：formularの長さが0でないとき例外を投げる
            assert(self.index == len(formula))

            #print("Formulareaderのcallのself.graphがセンテンスひとつのグラフ、GraphListのself.graphsがその集合リスト")
            graph_list.append(self.graph)

        return graph_list

    def extract_subformula(self):

        word = self.line[self.index]
        self.index += 1

        #tokenが-1（不明な語）のときUNKNOWN_CONST = 4をノードに加える．
        if word == -1: return self.graph.new_node(self.UNKNOWN_CONST)

        #self.vocabulary::self.reverse_vocabulary_indexをset_vocabで変換したもの
        meaning = self.vocabulary[word]

        #APPLICATION   = 0か、ABSTRACTION   = 1の場合
        if meaning == self.APPLICATION or meaning == self.ABSTRACTION:

            if meaning == self.ABSTRACTION:
                bvar = self.line[self.index]
                if bvar == -1 or self.vocabulary[bvar] != self.BOUNDED_VAR:
                    #抽象化の後に有界変数が続かない
                    raise Exception("Abstraction is not followed by a bounded variable")

            #self.graph.new_node:meanningの値をself.nodeに追加した後，self.nodeの長さ-1を返す
            result = self.graph.new_node(meaning)

            f1 = self.extract_subformula()
            f2 = self.extract_subformula()
            #edge_type=meaning, edge_nodes=(result,f1,f2)
            self.graph.add_edge(meaning, (result, f1, f2))

            #del:変数や要素の削除
            if meaning == self.ABSTRACTION: del self.variables[bvar]


            return result

        else:
            #FREE_VAR=2,BOUNDED_VAR=3の場合
            if meaning == self.FREE_VAR or meaning == self.BOUNDED_VAR:

                if meaning not in self.variables:
                    self.variables[word] = self.graph.new_node(meaning)
                return self.variables[word]

            else:
                result = self.graph.new_node(meaning)
                return result

class GraphList:

    def __init__(self, edge_arities, ver2 = False):
        self.graphs = []
        self.edge_arities = edge_arities
        #self.ver2 is True
        self.ver2 = ver2

        #足しあわせ
        #arities_cumsum=[3,6]
        arities_cumsum = np.cumsum(self.edge_arities)

        #arities_cumsum=[0,3,6]
        arities_cumsum = np.concatenate(([0], arities_cumsum))

        #self.edges_total=6
        self.edges_total = arities_cumsum[-1]

        #self.edge_indices=[0,3]
        self.edge_indices = arities_cumsum[:-1]

        #self.index_to_subarity=[2,2,2,2,2,2]
        self.index_to_subarity = np.concatenate([[arity]*arity for arity in self.edge_arities])-1

        #self.input_mul=13
        self.input_mul = sum(self.index_to_subarity)+1

        #self.subarity_zeros_arange=
        #[(2, array([0, 0]), array([0, 1])), (2, array([0, 0]), array([0, 1])), (2, array([0, 0]), array([0, 1])), (2, array([0, 0]), array([0, 1])), (2, array([0, 0]), array([0, 1])), (2, array([0, 0]), array([0, 1]))]
        self.subarity_zeros_arange = [
            (
                edge_subarity,
                np.zeros([edge_subarity], dtype = int),
                np.arange(edge_subarity)
            )
            for edge_subarity in self.index_to_subarity
        ]

    def append(self, graph):
        self.graphs.append(graph)

    def get_nodes(self):
        #print("get_nodes")
        if self.ver2: nodes_list = []
        else: nodes_list = [np.array([0])]
        nodes_list += [
            np.array(graph.nodes)
            for graph in self.graphs
        ]
        return np.concatenate(nodes_list)

    def get_conv_data(self):
        if self.ver2: return self.get_conv2_data()

        indices_list = [np.zeros(self.input_mul, dtype = int)]
        partition_list = [np.arange(self.input_mul)]
        partition_num = self.input_mul
        nodes_num = 1

        for graph in self.graphs:
            inputs = [
                [
                    [] for _ in range(self.edges_total)
                ]
                for node in graph.nodes
            ]

            # for [node][edge_type][position_in_edge] -> append list of neighbors
            #           [edge_type][position_in_edge] is flattened as [edge_index]
            #
            # inputs are of the shape
            #   [node][edge_index][* flexible *][arity-1]
            #
            for edge_type, edge_nodes in graph.edges:
                for i,node in enumerate(edge_nodes):
                    edge_index = self.edge_indices[edge_type] + i
                    neighbors = edge_nodes[:i]+edge_nodes[i+1:]
                    inputs[node][edge_index].append(neighbors)

            for node, node_inputs in enumerate(inputs):
                # indices to neighbors
                for neighbors, (subarity, zeros, partition) in zip(node_inputs, self.subarity_zeros_arange):

                    cur_mul = len(neighbors)
                    if cur_mul == 0:
                        indices_list.append(zeros)
                        cur_mul = 1
                    else:
                        indices_list.append(np.array(neighbors).transpose().flatten() + nodes_num)
                        partition = np.tile(partition.reshape([-1,1]), cur_mul).flatten()

                    partition_list.append(
                        partition + partition_num
                    )
                    partition_num += subarity

                # add the index to the node itself
                indices_list.append(np.array([node+nodes_num]))
                partition_list.append(np.array([partition_num]))
                partition_num += 1

            nodes_num += graph.nodes_num()

        gather_indices = np.concatenate(indices_list)
        #print(gather_indices)
        segments = np.concatenate(partition_list)
        #print("{} -> {} -> {}".format(nodes_num, partition_num, partition_num/13))

        return gather_indices, segments

    def get_conv2_data(self): # TODO: use_zero: True or False
        #print("get_conv2_data")
        nodes_num = 0
        result = [
            ([], []) for _ in range(self.edges_total)
        ]

        for graph in self.graphs:
            for edge_type, edge_nodes in graph.edges:
                for i,node in enumerate(edge_nodes):
                    edge_index = self.edge_indices[edge_type] + i
                    neighbors = np.array(edge_nodes[:i]+edge_nodes[i+1:])

                    #result[edge_index].append(
                    #    (neighbors + nodes_num, node + nodes_num)
                    #)
                    result[edge_index][0].append(neighbors + nodes_num)
                    result[edge_index][1].append(node + nodes_num)

            nodes_num += graph.nodes_num()

        result_np = []
        for data, subarity in zip(result, self.index_to_subarity):
            if len(data[0]) == 0:
                result_np.append((
                    np.zeros([0, subarity]),
                    np.zeros([0]),
                ))
            else:
                result_np.append((
                    np.stack(data[0]),
                    np.stack(data[1]),
                ))
                #print(np.max(result_np[-1][0]), np.max(result_np[-1][1]))

        return result_np

    def get_pool_data(self, remaining_layers):
        #print("get_pool_data")
        if self.ver2:
            partitions = []
            clusters_processed = 0
        else:
            partitions = [np.array([0])]
            clusters_processed = 1

        factor_graphs = []

        for graph in self.graphs:
            #import pymetis
            import simple_partition

            num_clusters \
                = int(np.ceil(graph.nodes_num()
                              ** (remaining_layers/float(remaining_layers+1))))

            #cut_size, partition \
            #    = pymetis.part_graph(num_clusters, adjacency=graph.to_simple_graph())
            partition \
                = simple_partition.part_graph(num_clusters, adjacency=graph.to_simple_graph())

            # rearrangement of the partition and discarding empty parts
            used_num = 0
            ori_to_used = [None]*num_clusters
            enhanced_partition = []
            for x in partition:
                if ori_to_used[x] is None:
                    ori_to_used[x] = used_num
                    used_num += 1
                enhanced_partition.append(ori_to_used[x])

            partition = enhanced_partition
            num_clusters = used_num

            # replace graph by its factorized version
            factor_graph = graph.partition(num_clusters, partition)
            factor_graphs.append(factor_graph)

            # add reindexed partition according to the whole list
            partition = np.array(partition)
            partitions.append(partition + clusters_processed)
            clusters_processed += num_clusters

        self.graphs = factor_graphs
        partition = np.concatenate(partitions)

        permutation = np.argsort(partition)
        #print(partition[permutation])
        #print("Pool {} -> {}".format(len(partition), clusters_processed))

        return permutation, partition[permutation]
