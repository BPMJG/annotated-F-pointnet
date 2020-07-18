# import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # train
# ROOT_DIR = os.path.dirname(BASE_DIR)
#
# q=True
# NUM_CHANNEL = 3 if q else 4
# print(NUM_CHANNEL)
# import importlib
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
# parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
# parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
# parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
# FLAGS = parser.parse_args()
#

# def import_module(name, package=None):
#     """Import a module.
#
#     The 'package' argument is required when performing a relative import. It
#     specifies the package to use as the anchor point from which to resolve the
#     relative import to an absolute import.
#
#     """
#     level = 0
#     if name.startswith('.'):
#         if not package:
#             msg = ("the 'package' argument is required to perform a relative "
#                    "import for {!r}")
#             raise TypeError(msg.format(name))
#         for character in name:
#             if character != '.':
#                 break
#             level += 1
#     print(_bootstrap._gcd_import(name[level:], package, level))
#     return _bootstrap._gcd_import(name[level:], package, level)
# MODEL = importlib.import_module(FLAGS.model)
# import time
# # 格式化成2016-03-20 11:45:39形式
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# file = r'C:\Users\11041\Desktop\word0309.docx'
# with open(file, 'a') as f:
#     f.write('this is my')

# path=r'C:\Users\11041\Desktop\pi.txt'
# with open(path,'w') as file_object:
#     file_object.write('3.1415926')
#
# a = 30.2,
# b = a + 1
# print(b)

# def add(x, y):
#     z = x + y
#     return z
#
# def sub(x, y):
#     z = x - y
#     return z
#
# def debug_test():
#     a = 10
#     b = 5
#     Sum = add(a, b)
#     Sub = sub(a, b)
#     print(Sum)
#     print(Sub)
#
# if __name__ == '__main__':
#     debug_test()
# import tensorflow as tf
# # batch_size = 32
# # size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size, ))
# # print(size_class_label_pl.shape)
# batch = tf.get_variable('batch', [],
#                         initializer=tf.constant_initializer(0), trainable=False)
# print(batch.shape)

# a = 5
# if str(type(a)) == "<class 'int'>":
#     print('yes')
# else:
#     print('no')
#
# print(type(a))
# print(type(type(a)))

# def is_prime_number(x):
#     flag = 1
#     for i in range(2, x):
#         if x % i == 0:
#             # print(str(x) + "不是质数")
#             return False
#             flag = 0
#             break
#         else:
#             pass
#
#     if flag == 1:
#         # print(str(x) + '是质数')
#         return True
# a=[]
# for x in range(3, 100):
#     if is_prime_number(x):
#         a.append(x)
# print(a)
# import numpy as np
# import tensorflow as tf
# sess=tf.Session()
# one_hot_vec = np.array([[3,5]])
# # global_feat=np.random.rand(32, 1, 1, 1024)
# # global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
# # global_feat_expand = tf.tile(global_feat, [1, 2048, 1, 1])
# print(one_hot_vec.shape)
# a = tf.expand_dims(one_hot_vec, 1)
# b=tf.tile(a,[2,3,4])
# print(a.shape)
# print(b.shape)
# print(sess.run(b))
# t = tf.constant([[[1, 1, 1], [2, 2, 2]],
#                  [[3, 3, 3], [4, 4, 4]],
#
#                  [[5, 5, 5], [6, 6, 6]]])
# print(t.shape)
# a=tf.slice(t, [2, 0, 0], [1, 1, 3])
# print(sess.run(a))
# mask_count=np.random.uniform(0, 3, (32,1,3))
# # point_cloud_xyz=np.random.rand(32, 2048,3)
# # mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1,1,3])*point_cloud_xyz,
# #                                   axis=2, keep_dims=True)
# # print(mask_xyz_mean.shape)
#
# max=tf.maximum(mask_count,1)
# print(sess.run(max))
# a = tf.constant([[1,2],
#                  [3,4]])
#
# b = tf.constant([[5,6],
#                  [7,8]])
# print(sess.run(a*b))
#(3,1,2)
# mask=tf.constant([[[1,2]],
#                   [[3,4]],
#                   [[5,6]]])
# print(mask.shape)
# logits=tf.constant([[[1,3],[2,4]],
#                   [[3,4],[4,1]],
#                   [[5,1],[6,3]]])
# print(mask.shape)
# q=sess.run(tf.squeeze(mask, axis=[2]))
# print(q)
#

# mask = tf.slice(logits,[0,0,0],[-1,-1,1]) < tf.slice(logits,[0,0,1],[-1,-1,1])
# # mask=tf.to_float(mask)
# # a=sess.run(mask)
# # print(a)
# # print(a.shape)
import numpy as np
# a=[]
# mask=np.array([[1.,1.,1.,1.,1.,1.],
#             [1.,0.,0.,0.,0.,0.],
#             [1.,0.,1.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.]])
# for i in range(len(mask)):
#     pos_indices = np.where(mask[i,:]>0.5)[0]
#     print(pos_indices)

# a=np.array([3,4,5,6,7,8])
# b=np.where(a>4)[0]
# print(b)

# choice1 = np.random.choice(30,20, replace=True)
# # print(choice1)
# # choice2 = np.concatenate((np.arange(30), choice1))
# # print(choice2)
indices = np.zeros((5,3, 2), dtype=np.int32)
pos=np.array([2,3,4,7,4,8])
choice=np.array([0,3,1])
for i in range(5):
    indices[i,:,1]= pos[choice]
    indices[i, :, 0] = i

print(indices)