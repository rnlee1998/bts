import torch
from torch import cuda 
import numpy as np
from torch._C import device
a1 = torch.tensor([1,1,1.],device=torch.device("cuda"))
a2 = torch.tensor([4,4,4.],device=torch.device("cuda"))
a3 = torch.tensor([0.1,2,2],device=torch.device("cuda"))

# a =[ [a1,a2,a3],[a2,a3,a1]]
# torch.tensor(a,device="cpu")
# print(np.sum(a))

arr = torch.tensor([-3.3475e+01, -3.0919e+01, -2.9631e+01, -3.6046e+01, -5.3513e+01,
        -7.2274e+01, -5.7512e+01, -7.0229e+01, -9.8621e+01, -7.0635e+01,
        -7.0336e+01, -2.8118e+01,  1.1704e+01,  2.0922e+00, -2.9106e+01,
         1.1632e+01, -3.5468e+01, -4.4599e+01, -6.8360e+01, -5.0318e+01,
        -1.0281e+02, -5.0215e+01, -7.2816e+01, -9.3646e+01, -1.2764e+02,
        -1.0397e+02, -6.4014e+01, -6.5104e+01, -9.1677e+01, -7.1063e+01,
        -7.5481e+01, -2.6433e+01, -3.8893e+01, -6.1764e+01, -8.1907e+01,
        -5.9348e+01, -5.5099e+01, -5.2780e+01, -1.5079e+01, -6.2041e+01,
        -1.9385e+00, -1.2007e+02,  8.2210e+00, -5.8251e+01, -1.6830e+01,
        -6.0766e+01, -8.4433e+00, -9.2487e+01, -4.1072e+01, -9.1372e+01,
        -1.5222e+01, -6.2728e+01, -2.2626e+01, -5.6825e+01, -2.9882e+01,
        -5.4070e+01, -6.0620e+01, -9.8263e+01, -6.9300e+01, -4.4652e+01,
        -3.3028e+01, -7.5551e+01, -3.7304e+01, -6.5097e+01, -5.3675e+01,
        -6.7573e+01, -4.7885e+01, -9.0864e+01, -5.5068e+01, -1.0511e+02,
        -2.9302e+01, -9.6063e+01, -2.6094e+01, -8.2225e+01, -6.4269e+00,
        -9.3053e+01, -2.2194e+01, -8.7752e+01,  6.2011e+00, -9.4919e+01,
        -2.5748e+01, -8.4346e+01, -2.0879e+01, -7.8302e+01,  1.0573e-01,
        -8.3869e+01, -4.6873e+01, -1.0259e+02, -7.8382e+01, -1.0929e+02,
        -6.6658e+01, -1.5895e+02, -4.7601e+01, -7.2240e+01, -1.6801e+01,
        -1.1238e+02, -6.7341e+01, -1.2568e+02, -2.4951e+01, -7.6883e+01,
        -6.6236e+01, -9.1854e+01, -8.0569e+01, -9.5054e+01, -1.3924e+02,
        -1.2891e+02,  1.1872e+01, -1.0573e+02, -9.9061e+01, -1.4276e+02,
        -9.5585e+02, -1.0078e+02, -6.4730e+01, -5.5198e+01, -2.5808e+01,
        -8.1250e+01, -4.9003e+01, -5.8711e+01, -7.6062e+01, -6.3182e+01,
        -5.9617e+01, -4.4728e+01, -7.0342e+01, -6.2140e+01, -7.8694e+00,
        -1.1441e+02, -3.3009e-01, -1.4282e+02,  5.1857e+01, -1.6574e+02,
         7.8905e+01, -1.5953e+02,  7.9293e+01, -1.4190e+02,  7.3914e+01,
        -1.5422e+02,  7.1898e+01, -1.9052e+02,  9.3002e+01, -1.7958e+02,
         1.0216e+02, -1.7630e+02,  9.3166e+01, -1.7439e+02,  9.1286e+01,
        -2.0202e+02,  1.0465e+02, -1.7698e+02,  9.3532e+01, -1.8192e+02,
         1.0422e+02, -2.0519e+02,  1.0025e+02, -2.0018e+02,  9.7891e+01,
        -1.7347e+02,  7.7173e+01, -1.8977e+02,  1.0373e+02,  2.8765e+00,
         5.1200e+02,  0.0000e+00,  2.1317e+01,  5.1089e+01,  2.5600e+02,
         0.0000e+00, -8.9473e+00,  2.5600e+02,  0.0000e+00, -1.4303e+01,
         2.5600e+02,  0.0000e+00,  4.9029e+00,  5.7600e+02,  0.0000e+00,
         3.4400e+00,  2.5600e+02,  0.0000e+00,  2.4718e+01,  7.0400e+02,
         0.0000e+00,  1.8510e+01,  2.5600e+02,  0.0000e+00,  5.1852e+00,
         8.3200e+02,  0.0000e+00,  3.0839e+01,  2.5600e+02,  0.0000e+00,
         9.9100e-02,  9.6000e+02,  0.0000e+00,  2.4183e+01,  2.5600e+02,
         0.0000e+00, -1.4915e+01, -8.7702e+00, -6.7244e+00,  8.2822e+00,
         3.7384e+00, -6.6503e+00,  2.2179e+00,  2.8580e-01, -2.0245e+00,
         1.2800e+02,  0.0000e+00,  1.8861e+01, -8.8183e+00, -4.9149e-01,
         1.9013e+00,  3.0427e+00, -1.7536e-01, -1.1301e+01,  6.4000e+01,
         0.0000e+00, -8.0801e+00,  2.1096e+00,  1.0009e+00,  4.9447e-01,
         4.0603e-01, -4.9509e+00, -3.0039e-01,  5.4358e-01, -1.1724e+00,
         5.0783e+00, -7.8621e-01])

# print(arr)
# ss = np.sum(arr,dim=0)
# sb = np.sum(arr,dim = 1)
# yu = torch.sum(arr)
# print(yu)
x = torch.tensor([[[128,110],
                                        [15,16]],
                                [[36,0.5],
                                [10,9]]])
low_feature, h_feature = x
print(low_feature)