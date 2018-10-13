from gazenet import GazeNet # the output python script of caffe2tensorflow
import tensorflow as tf
from tensorflow.python.tools import freeze_graph # tensorflow comes up with a tool allowing freeze graph

x_data = tf.placeholder(tf.float32, shape=[1, 227, 227, 3])
x_face = tf.placeholder(tf.float32, shape=[1, 227, 227, 3])
x_eyes_grid = tf.placeholder(tf.float32, shape=[1, 1, 169, 1])

net = GazeNet({'input_data': x_data, 'input_face': x_face, 'input_eyes_grid': x_eyes_grid})
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
net.load('gazenet.npy', sess)
saver = tf.train.Saver()
saver.save(sess, 'chkpt', global_step=0, latest_filename='chkpt_state')
tf.train.write_graph(sess.graph.as_graph_def(), './', 'gazenet.pb', False)

input_saver_def_path = ''
input_binary=True
input_checkpoint_path = 'chkpt-0'
input_graph_path = 'gazenet.pb'
output_graph_path = 'gazenet.pb'
output_node_names = 'prob'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
clear_devices = True

freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")
