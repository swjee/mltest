# 공유변수의 존재여부 체크 이미 존재시 재사용.
# 공유변수의 존재여부 체크 : get_variable()
# scope:   variable_scope(name_string, reuse =True or False )


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


logdir = "{}/var_test".format(root_logdir)
print(logdir)

print('------------------------------------')

with tf.variable_scope("my_scope"):
    x0 = tf.get_variable("x", shape=(), initializer=tf.constant_initializer(0.))
    x1 = tf.Variable(0., name="x")
    x2 = tf.Variable(0., name="x")

with tf.variable_scope("my_scope", reuse=True):
    x3 = tf.get_variable("x")
    x4 = tf.Variable(0., name="x")

with tf.variable_scope("", default_name="", reuse=True):
    x5 = tf.get_variable("my_scope/x")

print("x0:", x0.op.name)
print("x1:", x1.op.name)
print("x2:", x2.op.name)
print("x3:", x3.op.name)
print("x4:", x4.op.name)
print("x5:", x5.op.name)
print(x0 is x3 and x3 is x5 )



file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()

#init = tf.global_variables_initializer()
