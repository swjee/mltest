

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
#%matplotlib inline
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
f = x*x*y + y + 2

# log dir.. :: C:\MLTEST\handson-ml-master\tf_logs

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "C:/MLTEST/handson-ml-master/tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

print( logdir )


# namespace test...


reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                          # 책에는 없습니다.
        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # 책에는 없습니다.
        b = tf.Variable(0.0, name="bias")                             # 책에는 없습니다.
        z = tf.add(tf.matmul(X, w), b, name="z")                      # 책에는 없습니다.
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
            print("inside hass att")


        return tf.maximum(z, relu.threshold, name="max")  # 책에는 없습니다.
#threshold = tf.Variable(0.0, name="threshold")
# relu.......
logdir = "{}/relu".format(root_logdir)
print( logdir)
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relu.threshold = tf.Variable(19.0 , name = "threshold" )
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")



init = tf.global_variables_initializer()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()

#  run tf graph.
print('------------------------------------')

with tf.Session() as sess:
    sess.run(init)

    [output_val, relu_th_val ]  = sess.run([ output,relu.threshold ] , feed_dict={X:[[0,2,3]]})
    print('resultis ')
    print( output_val )
    print( relu_th_val)
    # relu.threshold = tf.Variable(30.0, name="threshold")
    op1 = tf.assign( relu.threshold , 30.0 )
    print('second..')
    sess.run(op1 )
    [output_val, relu_th_val] = sess.run([output, relu.threshold], feed_dict={X: [[0, 2, 3]]})
    print('resultis ')
    print(output_val)
    print(relu_th_val)

    [output_val, relu_th_val] = sess.run([output, relu.threshold], feed_dict={X: [[0, 2, 3]],relu.threshold:100.0})
    print('resultis ')
    print(output_val)
    print(relu_th_val)

