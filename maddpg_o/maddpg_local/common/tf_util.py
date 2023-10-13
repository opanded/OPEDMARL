import collections
import numpy as np
import os
import tensorflow as tf

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)
def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))
def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)
def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)
def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)

# ================================================================
# Misc
# ================================================================


def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0

# ================================================================
# 输入
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """通用的TensorFlow占位符。主要区别包括：
            - 可能在内部使用多个占位符, 并返回多个值
            - 可以对提供给占位符的值进行轻微后处理。
        """
        self.name = name

    def get(self):
        """返回表示可能经过后处理的占位符值的tf变量(们)。
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """给定数据, 将其输入到占位符(们)中。
        """
        raise NotImplemented()


class PlacholderTfInput(TfInput):
    def __init__(self, placeholder):
        """常规TensorFlow占位符的包装器。
        """
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


class BatchInput(PlacholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        """为给定形状和数据类型的张量批量创建一个占位符

        参数
        ----------
        shape: [int]
            批次中单个元素的形状
        dtype: tf.dtype
            用于张量内容的数字表示
        name: str
            底层占位符的名称
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))


class Uint8Input(PlacholderTfInput):
    def __init__(self, shape, name=None):
        """接受uint8格式的输入, 在传递给模型之前转换为float32并除以255。

        在GPU上, 这可以确保较低的数据传输时间。

        参数
        ----------
        shape: [int]
            张量的形状。
        name: str
            底层占位符的名称
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output


def ensure_tf_input(thing):
    """接受tf.placeholder或TfInput并输出等效的TfInput"""
    if isinstance(thing, TfInput):
        return thing
    elif is_placeholder(thing):
        return PlacholderTfInput(thing)
    else:
        raise ValueError("必须是占位符或TfInput")


# ================================================================
# 数学工具
# ================================================================

def huber_loss(x, delta=1.0):
    """参考：https://en.wikipedia.org/wiki/Huber_loss
    计算Huber损失
    """
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

# ================================================================
# 优化器工具
# ================================================================

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """使用`optimizer`最小化`objective`，关于`var_list`中的变量，
    同时确保每个变量的梯度范数被裁剪为`clip_val`
    """
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)

# ================================================================
# 全局会话
# ================================================================


SESS = None

def get_session():
    """返回最近创建的TensorFlow会话"""
    return SESS or tf.get_default_session()


def make_session(num_cpu):
    """返回一个仅使用 <num_cpu> 个CPU的会话"""
    # print("num_cpu:", num_cpu)
    tf_config = tf.ConfigProto(
        device_count={"CPU": 1},
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def set_session(sess):
    global SESS
    SESS = sess


def single_threaded_session():
    """返回一个仅使用单个CPU的会话"""
    return make_session(1)


ALREADY_INITIALIZED = set()


def initialize():
    """初始化全局范围内所有未初始化的变量"""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

# ================================================================
# 作用域
# ================================================================


def scope_vars(scope, trainable_only=False):
    """
    获取指定作用域内的变量
    作用域可以通过字符串来指定

    参数
    ----------
    scope: str 或 VariableScope
        变量所在的作用域。
    trainable_only: bool
        是否仅返回标记为可训练的变量。

    返回
    -------
    vars: [tf.Variable]
        在`scope`内的变量列表。
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """返回当前作用域的名称作为字符串，例如 deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """将父作用域名称附加到`relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name

# ================================================================
# 保存变量
# ================================================================


def load_state(fname, saver=None):
    """从位置<fname>加载所有变量到当前会话"""
    if saver is None:
        saver = tf.train.Saver()
    saver.restore(get_session(), fname+"model.ckpt")
    return saver

def load_params_from(fname, scope):
    import os
    vars = scope_vars(scope)
    # vars_dict = {var.name.split(":")[0]: var for var in vars}
    saver = tf.train.Saver(var_list=vars)
    saver.restore(get_session(), os.path.join(fname, "model.ckpt"))


def save_state(fname, saver=None):
    """将当前会话中的所有变量保存到位置<fname>"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # print('shabi')
    if saver is None:
        saver = tf.train.Saver()
    saver.save(get_session(), fname+"model.ckpt", meta_graph_suffix='meta', write_meta_graph=True, write_state=True)
    return saver

def save_variables(save_path, variables=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)

def load_variables(load_path, variables=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), '加载的变量数量与variables的长度不匹配'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)


# ================================================================
# 类似Theano的函数
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    """类似于Theano函数。接受一组基于TensorFlow占位符的占位符和表达式，
    并生成 f(inputs) -> outputs 的函数。函数 f 接受要提供给占位符的值，
    并生成输出中的表达式值。

    输入值可以按与输入相同的顺序传递，也可以基于占位符名称作为kwargs提供
    （通过构造函数传递或通过占位符.op.name访问）。

    示例：
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    参数
    ----------
    inputs: [tf.placeholder或TfInput]
        输入参数的列表
    outputs: [tf.Variable]或tf.Variable
        要从函数返回的输出或要返回的单个输出。返回的值也将具有相同的形状。
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput):
                assert len(inpt.op.inputs) == 0, "输入应该全部是rl_algs.common.TfInput的占位符"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan
        self.sess = get_session()
        # print(self.sess)

    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "提供了太多的参数"
        feed_dict = {}
        # 更新参数
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # 更新 kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                "此函数具有相同名称的两个参数 \"{}\"，因此无法使用kwargs。".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "缺少参数 " + inpt_name
        assert len(kwargs) == 0, "函数获得了额外的参数 " + str(list(kwargs.keys()))
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        # print("AA", self.sess, self.sess.graph, self.outputs_update)
        results = self.sess.run(self.outputs_update, feed_dict=feed_dict)[:-1]
        # print("BB")
        # results = []
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("检测到NaN")
        return results
