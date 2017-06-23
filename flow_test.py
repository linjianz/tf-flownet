import h5py
import cv2

# kitti_flow
# X_train: (41396, 128, 160, 3)
# X_val: (154, 128, 160, 3)
# X_test: (832, 128, 160, 3)

f1 = h5py.File('/media/csc105/Data/dataset/kitti_flow/X_val.hkl')
a = f1['data_0']
for i in range(5):
    cv2.imshow(('w%d' % i), a[i, :])
    cv2.waitKey()

f1.close()
