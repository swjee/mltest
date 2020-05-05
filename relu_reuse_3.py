# 공유변수의 존재여부 체크 이미 존재시 재사용.
# 공유변수의 존재여부 체크 : get_variable()
# scope:   variable_scope(name_string, reuse =True or False )
#  함수 외부에서 reuse 속성 세팅.


from __future__ import division, print_function, unicode_literals

# 공통
import numpy as np
import os


# 일관된 출력을 위해 유사난수 초기화
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# 맷플롯립 설정
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 그림을 저장할 폴더
PROJECT_ROOT_DIR = "/content/drive/My Drive/data/HML"
CHAPTER_ID = "tensorflow"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

reset_graph()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

# log dir.. :: C:\MLTEST\handson-ml-master\tf_logs

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "C:/MLTEST/handson-ml-master/tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

print(logdir)

# namespace test...

'''
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold",shape=(),initializer=tf.constant_initializer(0.0))

with tf.variable_scope("relu",reuse=True):
    threshold = tf.get_variable("threshold")

with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")
'''

reset_graph()


def relu(X):
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(30.0))
        w_shape = (int(X.get_shape()[1]), 1)  # 책에는 없습니다.
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # 책에는 없습니다.
        b = tf.Variable(0.0, name="bias")  # 책에는 없습니다.
        z = tf.add(tf.matmul(X, w), b, name="z")  # 책에는 없습니다.
        return tf.maximum(z, threshold, name="max")  # 책에는 없습니다.


# threshold = tf.Variable(0.0, name="threshold")
# relu.......
logdir = "{}/relu".format(root_logdir)
print(logdir)
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

# with tf.variable_scope("relu"):
#    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(30.0))


# with tf.variable_scope("", default_name="") as scope:
with tf.variable_scope("relu") as scope:
    first_relu = relu(X)     # 공유 변수를 만든 후
    scope.reuse_variables()  # 재사용합니다.
    relus = [first_relu] + [relu(X) for i in range(4)]

# relus = [relu(X) for i in range(2)]
output = tf.add_n(relus, name="output")

init = tf.global_variables_initializer()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()

#  run tf graph.
print('------------------------------------')

with tf.Session() as sess:
    sess.run(init)

    output_val = sess.run(output, feed_dict={X: [[0, 2, 3]]})
    print('resultis ')
    print(output_val)
    # relu.threshold = tf.Variable(30.0, name="threshold")