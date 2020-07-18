import numpy as np
# import tensorflow as tf
# sess=tf.Session()
# # init=tf.global_variables_initializer()
# # # # point_cloud=tf.Variable(tf.zeros([5,2,4]))
# point_cloud=tf.Variable(tf.zeros([3,3,4]),dtype=tf.float32)
# point_cloud_features = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])
# # # #         # 取出intensity
# point_cloud_stage1 = tf.concat([point_cloud, point_cloud_features], axis=-1)
# # sess.run(init)
# # # q=sess.run(point_cloud)
# # # print(q)
# # # v=tf.Variable(tf.zeros([3,3,3]))
# # # v2 = tf.Variable(tf.ones([10,5]))
# init=tf.global_variables_initializer()
# sess.run(init)
# q=sess.run(point_cloud_stage1)
# print(q)
# a=np.array([[[2,3,4],[5,2,9]],
#             [[2,3,4],[5,2,9]],
#             [[2,3,4],[5,2,9]]])
# print(a.size)
# print(len(a))
# one_hot_vec = np.zeros((3))
# print(one_hot_vec)
# cls_type= ['Car', 'Pedestrian', 'Cyclist']
# g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
# one_hot_vec = np.zeros((3))
# # one_hot_vec[g_type2onehotclass[cls_type]] = 1
# # for cls_type in g_type2onehotclass.values():
# #     print(cls_type)
#
# rotmat = np.array([[1, -1],
#                 [3, 2]])
# print(np.transpose(rotmat))
# choice = np.random.choice(10, 10, replace=False)
# print(choice)
# a=np.random.rand(1,3,1,4)
# print(a.shape)
# p=a.squeeze(2)
# print(p.shape)
a=np.random.randint(0,20,28)
print(a)
