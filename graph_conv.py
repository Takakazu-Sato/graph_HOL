import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.keras.layers import Input, Dense, Embedding
from tensorflow.keras.models import Model

class GraphConvPlaceHolder:

    def __init__(self, name = "graph_conv_data"):

        with tf.name_scope(name):
            self.gather_indices = tf.placeholder(tf.int32, [None], name = "gather_indices")
            self.segments = tf.placeholder(tf.int32, [None], name = "segments")

    def feed(self, conv_data):
        gather_indices, segments = conv_data
        return {self.gather_indices: gather_indices, self.segments: segments}

class GraphConv2PlaceHolder:

    def __init__(self, edge_arities, name = "graph_conv2_data"):

        with tf.name_scope(name):
            self.gather_scatter_indices = []
            for edge_index, edge_arity in enumerate(edge_arities):
                for node in range(edge_arity):
                    self.gather_scatter_indices.append((
                        tf.placeholder(tf.int32, [None, edge_arity-1],
                                       name = "gather_indices_{}_{}".format(edge_index, node)),
                        tf.placeholder(tf.int32, [None],
                                       name = "scatter_indices_{}_{}".format(edge_index, node)),
                    ))

    def feed(self, conv_data):
        result = dict()
        for ph, data in zip(self.gather_scatter_indices, conv_data):
            result[ph[0]] = data[0] # gather
            result[ph[1]] = data[1] # scatter

        return result

class GraphPoolPlaceHolder:

    def __init__(self, name = "graph_pool_data"):

        with tf.name_scope(name):
            self.permutation = tf.placeholder(tf.int32, [None], name = "permutation")
            self.segments = tf.placeholder(tf.int32, [None], name = "segments")

    # This method change the graph list to the partitioned version
    def feed(self, pool_data):
        permutation, segments = pool_data
        return {self.permutation: permutation, self.segments: segments}

class GraphPlaceHolder:

    def __init__(self, layer_num, edge_arities, ver2 = False, name = "graph_structure"):

        with tf.name_scope("graph_structure"):

            self.ver2 = ver2
            #[None]を与えると実行時に任意のサイズの配列を与えることが可能
            #1.Tensorの型 2.割り当てる変数のshape 3.変数の名前
            self.nodes = tf.placeholder(tf.int32, [None], name = "nodes")
            self.conv_data = []

            #layer_num=3
            #layer_num=2
            for i in range(layer_num):
                #ver2=True
                if ver2: self.conv_data.append(GraphConv2PlaceHolder(edge_arities, "conv_data{}".format(i+1)))
                else: self.conv_data.append(GraphConvPlaceHolder("conv_data{}".format(i+1)))
            #
            self.pool_data = [
                GraphPoolPlaceHolder("pool_data{}".format(i+1))
                for i in range(layer_num)
            ]
            self.index = 0
            self.layer_num = layer_num

    def get_nodes(self):

        return self.nodes

    def get_conv_data(self, index = None):
        if index is None:
            index = self.index
        return self.conv_data[index]

    def get_pool_data(self, index = None):
        if index is None:
            index = self.index
            self.index += 1
        return self.pool_data[index]

    def feed(self, graph_list):

        result = dict({self.nodes: graph_list.get_nodes()})
        for conv, pool, remaining_layers in zip(self.conv_data, self.pool_data,
                                               reversed(range(self.layer_num))):
            result.update(conv.feed(graph_list.get_conv_data()))
            result.update(pool.feed(graph_list.get_pool_data(remaining_layers)))


        return result

class GraphConvLayer:
    def __init__(self, output_dim, input_mul,
                 reduction = tf.segment_mean, activation_fn = tf.nn.relu):
        self.output_dim = output_dim
        self.input_mul = input_mul
        self.reduction = reduction
        self.activation_fn = activation_fn

    def __call__(self, structure, data): # data: [total_nodes, dim]
        #isinstance():ふたつの引数がオブジェクトとクラスまたはそのスーパークラスの関係にあればTrueを返す。
        if isinstance(structure, GraphPlaceHolder):
            assert(structure.ver2 == False)
            structure = structure.get_conv_data()

        input_dim = int(data.shape[-1])

        #gather:集約
        gathered = tf.gather(data, structure.gather_indices)
        print("gatherd:{}".format(gathered))

        """gathered = Embedding(
            output_dim=self.output_dim,
            input_dim=input_dim,
            )(gathered)"""

        #reduce:減らす
        reduced = self.reduction(gathered, structure.segments)
        print("reduced:{}".format(reduced))
        #arrange:整理、配置
        arranged = tf.reshape(reduced, [-1, self.input_mul*input_dim])
        print("arranged:{}".format(arranged))

        result = tf_layers.fully_connected(arranged, self.output_dim,
                                           activation_fn = self.activation_fn)
        return result

class GraphConv2Layer:
    def __init__(self, output_dim, edge_arities,
                 activation_fn = tf.nn.relu):
        self.output_dim = output_dim
        self.edge_arities = edge_arities
        self.activation_fn = activation_fn
        self.index_to_subarity = np.concatenate([[arity]*arity for arity in edge_arities])-1

        print("output_dim: {}".format(self.output_dim))
        print("edge_arities: {}".format(self.edge_arities))
        print("index_to_suarity: {}".format(self.index_to_subarity))

    def __call__(self, structure, data): # data: [total_nodes, dim]

        if isinstance(structure, GraphPlaceHolder):
            assert(structure.ver2 == True)
            structure = structure.get_conv_data()

        input_dim = int(data.shape[-1])

        #tf.contrib.layers.fully_connected(,activation fn=None)
        next_data = tf_layers.linear(data, num_outputs = self.output_dim)

        """next_data = Embedding(
            output_dim=self.output_dim,
            input_dim=input_dim,
            )(next_data)"""

        #テンソルの形状
        shape = tf.shape(next_data)

        #tf.expand_dims(input, axis) は input テンソルの axis に指定した階に 1 次元挿入する．
        #e.g. shape が (2,3,4) の input の axis=2 に 1 次元を挿入すると (2,3,1,4) になる．
        #tf.gather(params, indices, axis) は params テンソルの axis に指定した階でスライスして，indeices で指定したインデックスのテンソルだけ取り出してくっつける．
        #e.g. [[1,2,3],[4,5,6]] という params テンソルの axis=1 でスライスすると [1,4] と [2,5] と [3,6] という 3 つのテンソルができ，indeces=[1,2] なら 1,2 番目のテンソルを取り出してくっつけるので [[2,3],[5,6]] になる．

        for (gather_indices, scatter_indices), subarity \
            in zip(structure.gather_scatter_indices, self.index_to_subarity):
            gathered = tf.gather(data, gather_indices)
            #reshape:形状を変える
            flattened = tf.reshape(gathered, [-1, subarity*input_dim])
            #変換する
            transformed = tf_layers.linear(flattened, num_outputs = self.output_dim)
            #next_data = tf.scatter_nd_add(next_data, scatter_indices, transformed)
            scatter_indices = tf.expand_dims(scatter_indices, axis = -1)
            next_data = next_data + tf.scatter_nd(scatter_indices, transformed, shape)

        print(next_data)
            #activation_fn=tf.nn.relu
        return self.activation_fn(next_data)

class GraphPoolLayer:
    def __init__(self, reduction = tf.segment_max):
        self.reduction = reduction

    def __call__(self, structure, data):
        if isinstance(structure, GraphPlaceHolder):
            structure = structure.get_pool_data()

        permuted = tf.gather(data, structure.permutation)
        result = self.reduction(permuted, structure.segments)
        return result

class GraphInitLayer:

    def __init__(self, dim, vocab_size):
        self.dim = dim
        self.vocab_size = vocab_size

    def __call__(self, structure):

        embeddings = tf.concat([
            tf.zeros([1, self.dim]),
            tf.get_variable(name="embeddings", shape=[self.vocab_size, self.dim]),
        ], axis = 0)

        result = tf.gather(embeddings, structure.get_nodes())

        return result

class ConvNetwork:#Tensorboardの表示，レイヤー員sタンスの呼び出し.

    # layer_signature = ( (2, 64), (2, 128), (2, 256) ): (l,d) a = number of layers, b = dimension between poolings
    #def __init__(self, vocab_size, layer_signature, edge_arities, ver2 = False):
    def __init__(self, vocab_size,layer_signature, edge_arities, ver2 ):

        self.ver2 = ver2
        print("ver2:{}".format(ver2))
        #listよりnumpy配列のほうが早く計算できる．
        #self.edge_arities=[3,3]
        self.edge_arities = np.array(edge_arities)
        #input_mul=13
        input_mul = np.sum(self.edge_arities * (self.edge_arities-1)) + 1

        self.placeholder = None
        self.layer_signature = layer_signature

        #Tensor("Step/layer_0_1_conv/add:0", shape=(?, 64), dtype=float32)
        self.layers = [
            tf.make_template('layer_0_0_init', GraphInitLayer(layer_signature[0][1], vocab_size))
        ]

        for i, (layer_num, dim) in enumerate(layer_signature):
            for j in range(layer_num):
                if ver2:
                    #GraphConv2Layerインスタンス生成
                    print("ver2:{}".format(ver2))
                    layer = GraphConv2Layer(dim, edge_arities)
                else:
                    layer = GraphConvLayer(dim, input_mul)
                name = "layer_{}_{}_conv".format(i, j+1)

                self.layers.append(tf.make_template(name, layer))

            self.layers.append(tf.make_template("layer_{}_pool".format(i+1),
                                                GraphPoolLayer()))
            print("layer_{}_pool".format(i+1))

    def __call__(self):
        data = None

        if self.placeholder is not None:
            #convnetworkは１度だけ呼び出すことができます
            raise Exception("ConvNetwork can be called just once")

        #self.layer_signature=2
        self.placeholder = GraphPlaceHolder(layer_num = len(self.layer_signature),
                                            edge_arities = self.edge_arities,
                                            ver2 = self.ver2)

        for layer in self.layers:
            #data is None
            #layer=GraphConv2Layerインスタンスのcallを呼び出し.
            if data is None:
                print("data is None!!!!!!!!!!")
                data = layer(self.placeholder)

            else:
                print("dataは存在!!!!!!!!!!")
                data = layer(self.placeholder, data)


        #self.ver2=True
        if self.ver2: return data
        else: return data[1:]

    def feed(self, graph_list):

        return self.placeholder.feed(graph_list)
