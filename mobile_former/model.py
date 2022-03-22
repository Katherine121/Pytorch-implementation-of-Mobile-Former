import torch
import torch.nn as nn

from torch.nn import init
import torch.nn.functional as F

from mobile_former.mobile import Mobile, MobileDown
from mobile_former.former import Former
from mobile_former.bridge import Former2Mobile


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(torch.add(x, 3.0)) / 6.0
        return out


class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, stride)
        else:
            self.mobile = Mobile(3, inp, exp, out, stride)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, x, z):
        z_out = self.former(z)
        x_hid = self.mobile(x)
        x_out = self.former2mobile(x_hid, z_out)
        return x_out, z_out


class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        # 初始化6*192token
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))

        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish()
        )
        # bneck 先*2后还原，步长为1，组卷积
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], kernel_size=3, stride=cfg['bneck']['s'], padding=1,
                      groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            # 把{'inp': 12, 'exp': 72, 'out': 16, 'stride': 2, 'heads': 2}和token维度192传进去
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))

        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))

        self.head1 = nn.Linear(exp + cfg['embed'], cfg['fc1'])
        self.tanh = hswish()
        self.head2 = nn.Linear(cfg['fc1'], cfg['fc2'])

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # batch_size
        b = x.shape[0]
        # 因为最开始初始化的是1*6*192，在0维度重复b次，1维度重复1次，2维度重复1次，就形成了b*6*192
        z = self.token.repeat(b, 1, 1)
        # token = torch.Tensor([[[-7.0548e-03, -1.4787e-04, -6.3539e-03, -3.3052e-03, 2.0736e-04,
        #                     5.4176e-03, -1.1272e-03, 1.2659e-03, 1.2475e-03, -5.7435e-03,
        #                     -8.8878e-03, -8.0723e-04, -1.0396e-02, -4.1826e-03, -5.8538e-03,
        #                     6.2607e-03, -2.3034e-02, -1.0841e-02, 4.6107e-05, -5.2068e-03,
        #                     -6.0257e-03, 6.5824e-05, -4.8144e-03, -4.4620e-03, -3.7831e-03,
        #                     -1.5922e-03, -9.7508e-03, 4.2080e-03, -4.4906e-03, -3.4348e-03,
        #                     -2.4479e-03, -4.8947e-04, -4.1292e-03, -2.8959e-02, -4.4884e-03,
        #                     6.5331e-03, -9.6041e-03, -1.2215e-02, -2.0039e-03, -8.6193e-03,
        #                     -6.7541e-04, 7.5982e-04, -2.0436e-02, -1.1817e-03, -7.8008e-03,
        #                     1.3052e-03, -4.4150e-03, -2.1062e-03, -5.1725e-03, -1.3263e-03,
        #                     2.6756e-03, 1.7469e-03, -1.0756e-02, -1.1706e-04, -8.4620e-03,
        #                     -1.5846e-03, 1.2556e-03, -4.6905e-03, -8.0230e-03, -7.1606e-03,
        #                     -3.8625e-03, -6.4675e-03, -9.4311e-03, -6.7213e-03, -2.0526e-03,
        #                     -1.7530e-03, -2.3654e-03, -2.5682e-03, 3.9032e-03, -3.7361e-03,
        #                     -6.7834e-03, -1.9786e-03, -4.6990e-03, 7.7936e-03, 2.1806e-04,
        #                     -5.3878e-03, -4.3086e-03, 1.0892e-03, 5.6415e-03, -5.0733e-03,
        #                     -1.3536e-03, 6.9842e-04, -7.6700e-03, -5.5178e-03, 3.7745e-03,
        #                     -6.8866e-03, -2.5275e-03, -2.2500e-03, -9.3372e-03, 6.1555e-03,
        #                     7.8020e-03, -4.3444e-05, -2.1304e-03, -2.6447e-03, -8.0282e-03,
        #                     2.2914e-03, -6.8329e-04, 2.4414e-03, -3.8896e-03, -3.2848e-03,
        #                     -5.6255e-03, -2.5096e-03, -6.7666e-03, 2.6100e-03, -6.4053e-03,
        #                     -1.8074e-03, -2.9047e-03, -4.4761e-03, -5.4843e-03, -1.6027e-03,
        #                     -5.7049e-04, -7.7090e-03, -8.3108e-03, -6.8657e-03, 3.7249e-04,
        #                     -3.9567e-03, 2.8218e-01, -2.0726e-03, -7.7906e-03, 2.0460e-03,
        #                     9.3965e-04, -3.2554e-03, -7.8031e-03, 3.5271e-03, -3.0605e-03,
        #                     -1.3198e-02, -1.2045e-02, -4.6395e-03, -6.4210e-03, -7.0338e-03,
        #                     1.7386e-03, 2.0885e-03, -1.0197e-02, -1.9753e-03, -8.2597e-03,
        #                     -3.1333e-03, 3.2518e-04, -4.4163e-03, 1.2353e-03, 1.3111e-02,
        #                     -1.8928e-04, -3.1084e-03, -1.3500e-04, -2.0569e-04, -6.2121e-04,
        #                     -2.1985e-03, 1.8796e-03, -5.1774e-03, -4.0560e-03, 5.2277e-03,
        #                     -4.9109e-03, -1.2892e-03, 5.8551e-03, 1.0537e-03, -7.3860e-04,
        #                     -4.7651e-03, -1.3329e-04, 3.5654e-03, -1.2728e-02, 1.9785e-03,
        #                     -1.2494e-03, -4.1503e-03, 3.6579e-05, -3.1829e-03, 5.4027e-03,
        #                     1.7812e-03, 2.3084e-03, -1.2901e-02, -1.0155e-02, 5.8997e-03,
        #                     -5.7371e-03, 1.7757e-03, 1.0183e-02, -3.5013e-03, -2.2847e-03,
        #                     -3.4206e-03, -4.0648e-03, -1.6196e-02, -7.3560e-03, 6.7360e-04,
        #                     -4.2434e-03, -3.7249e-03, -3.8137e-03, -1.3870e-02, 2.0769e-03,
        #                     -3.3085e-03, 2.0577e-03, 2.6179e-03, 1.1118e-02, 1.0646e-03,
        #                     -6.5438e-03, -4.1004e-03],
        #                    [3.9412e-02, 4.6605e-02, 4.4226e-02, 5.5100e-02, 3.3365e-02,
        #                     3.0941e-02, 4.3950e-02, 4.6866e-02, 4.4539e-02, 4.3670e-02,
        #                     3.5825e-02, 3.7991e-02, 4.9647e-02, 5.2947e-02, 4.3621e-02,
        #                     5.5147e-02, 7.0903e-02, 5.4699e-02, 5.2195e-02, 3.4190e-02,
        #                     5.6208e-02, 4.7562e-02, 5.5986e-02, 5.0749e-02, 5.1164e-02,
        #                     4.4148e-02, 5.3190e-02, 3.9358e-02, 4.0089e-02, 6.2585e-02,
        #                     4.5395e-02, 3.4952e-02, 4.0288e-02, -1.0673e-01, 2.8327e-02,
        #                     5.1626e-02, 5.3633e-02, 4.3180e-02, 4.8717e-02, 4.8028e-02,
        #                     3.8081e-02, 5.3803e-02, 6.8205e-02, 3.7599e-02, 3.5603e-02,
        #                     4.6897e-02, 5.4169e-02, 4.4848e-02, 3.4377e-02, 3.5796e-02,
        #                     5.3026e-02, 5.0453e-02, 4.6468e-02, 4.8236e-02, 4.6252e-02,
        #                     3.6408e-02, 4.1310e-02, 5.3542e-02, 4.3213e-02, 4.6180e-02,
        #                     4.4034e-02, 4.4744e-02, 3.9525e-02, 3.9267e-02, 4.8050e-02,
        #                     3.9374e-02, 3.8884e-02, 5.1543e-02, 5.1024e-02, 5.7782e-02,
        #                     5.4338e-02, 4.3063e-02, 4.2745e-02, 3.4694e-02, 3.6927e-02,
        #                     4.5745e-02, 4.1386e-02, 4.0114e-02, 5.3486e-02, 4.1019e-02,
        #                     4.1427e-02, 3.6186e-02, 4.7183e-02, 4.2063e-02, 3.5009e-02,
        #                     4.7520e-02, 4.4950e-02, 5.1132e-02, 5.1567e-02, 3.2456e-02,
        #                     5.1035e-02, 4.4586e-02, 5.6380e-02, 5.0852e-02, 4.2066e-02,
        #                     5.4981e-02, 4.1874e-02, 3.9314e-02, 2.5856e-02, 5.6079e-02,
        #                     4.7116e-02, 5.3767e-02, 3.2518e-02, 4.1648e-02, 5.2073e-02,
        #                     4.3399e-02, 4.3493e-02, 4.4162e-02, 6.1432e-02, 3.7870e-02,
        #                     4.7615e-02, 5.6790e-02, 2.2701e-02, 4.7673e-02, 5.5377e-02,
        #                     4.5501e-02, 4.6034e-01, 3.2209e-02, 3.6020e-02, 5.9228e-02,
        #                     6.0804e-02, 4.3987e-02, 3.7095e-02, 3.9341e-02, 4.7599e-02,
        #                     5.0930e-02, 5.0681e-02, 3.1620e-02, 4.5514e-02, 4.2476e-02,
        #                     4.9406e-02, 4.1482e-02, 5.5845e-02, 4.6477e-02, 4.0031e-02,
        #                     5.8369e-02, 5.2847e-02, 2.7728e-02, 4.8842e-02, -3.1026e-02,
        #                     4.7690e-02, 4.8285e-02, 3.5266e-02, 4.1188e-02, 4.0884e-02,
        #                     4.2354e-02, 4.3363e-02, 4.5391e-02, 5.5630e-02, 2.2099e-02,
        #                     5.2128e-02, 5.7857e-02, 3.7080e-02, 3.3151e-02, 5.1021e-02,
        #                     3.4342e-02, 4.7466e-02, 4.4289e-02, 3.2008e-02, 3.8064e-02,
        #                     4.7879e-02, 5.3393e-02, 5.1548e-02, 4.0164e-02, 4.6784e-02,
        #                     3.7616e-02, 4.0303e-02, 6.4582e-02, 3.6262e-02, 4.8053e-02,
        #                     6.5421e-02, 5.9045e-02, 5.3291e-02, 5.1781e-02, 4.2931e-02,
        #                     5.1023e-02, 4.0675e-02, 4.7287e-02, 4.1910e-02, 4.2472e-02,
        #                     5.0775e-02, 4.2488e-02, 3.7280e-02, 4.2600e-02, 4.4712e-02,
        #                     4.0806e-02, 3.2714e-02, 3.8347e-02, 3.6318e-02, 5.3734e-02,
        #                     5.0583e-02, 6.0563e-02],
        #                    [-3.9481e-02, -3.7972e-02, -4.3401e-02, -4.8046e-02, -3.4255e-02,
        #                     -3.4841e-02, -3.8420e-02, -2.8254e-02, -4.5176e-02, -2.5439e-02,
        #                     -4.4708e-02, -3.0413e-02, -4.6351e-02, -3.2337e-02, -3.7228e-02,
        #                     -3.6169e-02, -6.7723e-02, -4.5312e-02, -3.5033e-02, -3.5616e-02,
        #                     -3.3454e-02, -4.2763e-02, -4.1506e-02, -4.0817e-02, -3.7626e-02,
        #                     -3.5362e-02, -4.5803e-02, -3.5168e-02, -4.4455e-02, -4.0322e-02,
        #                     -3.6171e-02, -2.9261e-02, -3.8194e-02, 1.3212e-03, -3.9902e-02,
        #                     -3.2822e-02, -3.6121e-02, -4.3349e-02, -4.1864e-02, -3.9211e-02,
        #                     -3.8131e-02, -3.7268e-02, -3.4628e-01, -4.1778e-02, -3.8473e-02,
        #                     -4.5906e-02, -4.3029e-02, -3.1370e-02, -3.6979e-02, -3.2019e-02,
        #                     -4.2190e-02, -4.2197e-02, -4.2552e-02, -4.2534e-02, -3.8593e-02,
        #                     -3.7046e-02, -2.9957e-02, -4.2067e-02, -4.3873e-02, -4.3692e-02,
        #                     -3.4921e-02, -4.1655e-02, -3.8590e-02, -3.8165e-02, -3.7488e-02,
        #                     -3.4977e-02, -3.5636e-02, -3.6947e-02, -4.5206e-02, -3.4604e-02,
        #                     -5.0553e-02, -4.3908e-02, -3.3666e-02, -3.3068e-02, -3.8151e-02,
        #                     -4.7620e-02, -3.9082e-02, -3.7147e-02, -2.8498e-02, -3.6693e-02,
        #                     -3.5214e-02, -3.5427e-02, -3.4255e-02, -3.8834e-02, -3.6941e-02,
        #                     -4.4380e-02, -4.5821e-02, -3.6857e-02, -2.5304e-02, -2.6998e-02,
        #                     -3.9863e-02, -4.0011e-02, -4.2033e-02, -2.7422e-02, -3.7377e-02,
        #                     -3.6890e-02, -2.9795e-02, -2.7691e-02, -3.8565e-02, -3.3979e-02,
        #                     -3.9984e-02, -4.5776e-02, -3.9310e-02, -4.2762e-02, -3.1982e-02,
        #                     -3.5834e-02, -3.4589e-02, -4.4022e-02, -3.5640e-02, -3.7584e-02,
        #                     -3.3878e-02, -4.4458e-02, -3.8850e-02, -3.6738e-02, -4.1872e-02,
        #                     -3.8892e-02, -1.6512e-02, -3.9568e-02, -4.0549e-02, -3.5396e-02,
        #                     -4.5787e-02, -3.5538e-02, -4.1585e-02, -2.8817e-02, -2.7719e-02,
        #                     -3.6737e-02, -3.8369e-02, -3.7098e-02, -3.7212e-02, -3.6574e-02,
        #                     -3.9311e-02, -3.9966e-02, -3.7876e-02, -4.1813e-02, -3.3911e-02,
        #                     -5.0937e-02, -3.9486e-02, -4.7443e-02, -3.7963e-02, 2.8073e-02,
        #                     -3.1140e-02, -3.6666e-02, -3.0297e-02, -3.6094e-02, -4.1801e-02,
        #                     -4.0538e-02, -4.2317e-02, -3.9283e-02, -3.9168e-02, -3.3032e-02,
        #                     -3.7619e-02, -5.3180e-02, -2.9760e-02, -4.2388e-02, -4.0077e-02,
        #                     -4.6375e-02, -3.8592e-02, -4.2658e-02, -4.3283e-02, -3.3591e-02,
        #                     -3.5455e-02, -3.3110e-02, -3.4572e-02, -2.5951e-02, -3.7756e-02,
        #                     -3.7778e-02, -3.5009e-02, -3.9003e-02, -4.9944e-02, -3.5326e-02,
        #                     -3.8513e-02, -3.4711e-02, -2.8414e-02, -4.1073e-02, -3.6326e-02,
        #                     -4.8504e-02, -3.5434e-02, -3.8831e-02, -4.1741e-02, -3.5057e-02,
        #                     -3.9320e-02, -4.0630e-02, -3.6765e-02, -4.1859e-02, -3.9533e-02,
        #                     -4.4602e-02, -4.1445e-02, -3.7854e-02, -3.2628e-02, -4.0352e-02,
        #                     -4.3984e-02, -4.1888e-02],
        #                    [-9.9407e-03, -1.4115e-03, -6.9938e-03, 7.9003e-03, -1.3997e-02,
        #                     -5.6554e-03, -8.0150e-03, -9.5252e-03, -1.6770e-02, -7.0578e-03,
        #                     -1.7571e-02, 4.1757e-03, -6.3428e-03, -2.1324e-02, -1.9838e-03,
        #                     8.8310e-04, 1.4368e-02, -6.4280e-03, -1.2659e-02, 3.8956e-03,
        #                     -6.6280e-03, -5.8218e-03, -5.5798e-03, -5.9263e-03, -8.5326e-04,
        #                     -8.4485e-03, -1.5892e-02, -7.1654e-03, -8.2011e-03, 3.6886e-03,
        #                     -7.5211e-03, -2.4625e-03, -1.6698e-02, -3.7385e-02, -2.4631e-02,
        #                     -5.2953e-03, -4.3466e-03, -3.4595e-03, -8.4607e-03, -7.6626e-03,
        #                     -9.1499e-03, -1.0318e-02, -4.6015e-02, -9.9765e-03, -1.0647e-02,
        #                     -1.1814e-02, -6.8577e-03, -6.3805e-03, -1.4725e-02, -1.3758e-02,
        #                     -2.0932e-03, -5.5496e-03, -1.5982e-02, -3.4237e-03, -1.1983e-02,
        #                     -6.0960e-03, -6.3102e-03, -4.0011e-03, -1.6600e-02, -6.1203e-03,
        #                     -1.8782e-03, -1.1034e-02, -1.6329e-02, 1.4601e-03, -1.0691e-03,
        #                     -1.0049e-02, -2.3922e-02, 1.5658e-03, 2.7403e-03, -9.4732e-03,
        #                     -7.4739e-03, -1.2053e-02, -1.8561e-02, -7.5180e-03, -3.5770e-03,
        #                     -3.2365e-04, -9.0371e-03, -1.3311e-02, -5.2443e-03, -6.8623e-03,
        #                     -8.9147e-03, 5.9759e-05, -7.1884e-03, -1.9319e-02, -1.1498e-02,
        #                     -5.0179e-03, -3.6069e-03, -9.4847e-03, -8.7954e-03, -9.2914e-03,
        #                     -4.1749e-03, -7.4541e-04, -1.6338e-02, -6.9121e-03, -8.8095e-04,
        #                     -8.9325e-03, -1.0655e-02, -6.4734e-03, -3.8470e-03, -1.3236e-02,
        #                     -8.3131e-03, -3.6695e-03, 2.1692e-03, -6.8598e-03, -1.3186e-02,
        #                     -6.4601e-03, 2.2332e-03, -8.0641e-03, -1.0632e-02, -1.7579e-02,
        #                     -6.7013e-03, -7.9902e-03, -9.8189e-03, -8.9317e-03, -8.2219e-03,
        #                     -7.2846e-03, 2.8514e-01, -9.3179e-03, -1.8933e-02, -6.0262e-03,
        #                     -5.8612e-03, -2.5726e-03, -1.3638e-02, -2.4181e-03, 3.0029e-03,
        #                     4.9397e-04, -1.4074e-02, -9.1681e-03, -1.2010e-02, -7.5199e-03,
        #                     -2.5226e-03, -1.3083e-02, -1.5445e-02, -1.5942e-02, -1.4499e-02,
        #                     -1.4872e-02, -2.7243e-03, -2.4156e-02, -5.2614e-03, -5.8551e-02,
        #                     -3.1838e-04, -6.9915e-03, -6.0345e-03, -8.4223e-03, 1.2337e-06,
        #                     -5.6256e-03, -6.2454e-03, -2.3503e-03, -2.0622e-02, -1.3318e-02,
        #                     -1.1374e-02, -7.8563e-03, -1.5910e-03, -1.1557e-02, -1.8914e-02,
        #                     -1.2281e-02, -3.4388e-03, -7.4428e-03, -1.0336e-02, -1.4295e-02,
        #                     -1.4970e-02, -4.5557e-03, -3.1948e-03, -4.7673e-03, -5.6239e-03,
        #                     -1.3553e-02, -7.7409e-03, -9.4607e-03, -1.8610e-02, -8.9389e-03,
        #                     -1.0089e-02, -1.4453e-03, 1.4531e-03, -6.8850e-04, 1.7233e-03,
        #                     -5.2745e-04, -8.6040e-03, -9.5674e-03, -8.5040e-03, -1.2596e-02,
        #                     -7.9758e-03, -1.2024e-02, -8.6450e-03, -1.1188e-02, 5.7452e-03,
        #                     -1.1493e-02, -5.7370e-03, 6.9424e-03, 4.4869e-03, -1.8864e-02,
        #                     -3.7993e-03, -1.4067e-02],
        #                    [1.2557e-02, 1.2576e-02, 2.2853e-02, -3.5131e-02, 6.9686e-02,
        #                     1.5650e-02, -6.1485e-03, 2.0246e-02, 1.9459e-02, 1.2931e-02,
        #                     1.8231e-02, 1.8416e-02, 1.8993e-02, 1.5048e-02, 3.8852e-02,
        #                     6.8661e-03, -1.5767e-01, 1.5418e-02, 1.6841e-02, 2.6204e-02,
        #                     3.4025e-02, -2.2390e-02, 1.8658e-02, 2.3445e-02, 1.3707e-02,
        #                     4.0083e-02, 3.1400e-02, 1.0850e-02, 1.1051e-02, 7.8442e-03,
        #                     1.5894e-02, 1.6950e-02, 1.7679e-02, -2.5012e-03, 5.9588e-02,
        #                     4.6807e-02, 2.0241e-02, 2.3068e-02, 3.1201e-02, 2.4359e-02,
        #                     9.4498e-03, 1.6005e-02, 1.2280e-02, 2.0354e-02, 2.3785e-02,
        #                     2.7575e-02, 1.4437e-02, 2.0319e-02, 9.9938e-03, 5.5434e-02,
        #                     2.4812e-02, -5.6517e-04, -2.5349e-02, 3.9900e-03, 1.7391e-02,
        #                     -2.5362e-02, 2.2528e-02, 2.4456e-02, 1.0253e-02, 4.6689e-03,
        #                     1.6252e-02, 3.0221e-02, 2.7131e-02, 1.7633e-02, -1.2475e-02,
        #                     -5.5534e-03, 4.9265e-02, -1.7771e-04, 1.0291e-02, 7.2147e-03,
        #                     -3.6455e-03, 1.6995e-02, 4.5811e-02, -2.5720e-02, 3.0223e-02,
        #                     -1.9810e-02, 3.9605e-02, 3.6734e-02, 4.1007e-02, 1.5660e-02,
        #                     1.9160e-02, 1.9330e-02, 2.6021e-02, 3.5280e-02, 6.4512e-02,
        #                     1.7387e-02, 3.9181e-03, -8.0421e-04, 2.5046e-03, 1.7377e-02,
        #                     1.5272e-02, 1.2886e-02, 6.0391e-03, 2.0839e-02, 1.3412e-02,
        #                     3.2313e-02, 5.2415e-02, 2.4330e-02, 1.3857e-02, 1.9325e-02,
        #                     2.4892e-02, 1.6694e-02, 1.2925e-02, -2.6194e-02, 1.7210e-02,
        #                     3.1016e-03, 2.8697e-02, 3.5474e-02, 1.4486e-02, 1.7928e-02,
        #                     1.4137e-02, -9.0278e-03, 1.5117e-03, 1.9855e-02, 1.6163e-02,
        #                     2.0123e-02, 5.3799e-03, 5.1669e-03, 2.0227e-02, 1.4680e-02,
        #                     2.9761e-02, 1.0342e-02, 3.1447e-02, 1.1718e-02, 2.8733e-02,
        #                     -1.1163e-02, 2.7785e-02, 1.6015e-02, 1.3015e-02, 2.9640e-02,
        #                     3.1219e-02, 2.0851e-02, 1.0622e-02, 2.6715e-02, 1.0814e-02,
        #                     1.8825e-02, -3.4809e-02, 1.1678e-02, 1.8894e-02, 3.1539e-01,
        #                     2.0229e-02, 1.3041e-02, 2.9312e-02, 5.5497e-03, 1.8552e-02,
        #                     9.3512e-03, 2.9080e-02, 1.8697e-02, 1.8131e-02, 1.6777e-02,
        #                     3.3788e-03, 1.6258e-02, 2.0621e-02, 2.7396e-02, 8.6652e-03,
        #                     6.8821e-03, -2.9206e-02, 2.5995e-02, 3.8944e-03, 4.4247e-02,
        #                     2.8697e-02, 1.9967e-02, -1.4459e-02, 2.2915e-02, -1.0967e-02,
        #                     7.4455e-03, 1.1550e-02, 1.2180e-02, 1.6162e-02, 3.2819e-02,
        #                     2.6718e-02, 1.9112e-02, 1.7321e-02, 1.8644e-02, 3.9133e-02,
        #                     1.4031e-02, 2.9304e-02, 9.5824e-03, 1.1423e-02, 3.5398e-02,
        #                     2.2063e-02, 2.0204e-02, 1.2427e-02, 1.0517e-02, 1.6984e-02,
        #                     2.5051e-02, 1.6868e-02, -2.0471e-02, 8.2735e-03, 3.2532e-02,
        #                     1.4130e-02, 1.7373e-02],
        #                    [-1.5437e-02, -1.1299e-03, -2.0369e-02, -2.6772e-03, -3.2092e-02,
        #                     -5.6139e-03, -1.8777e-03, -1.7574e-02, -4.6916e-03, -1.4651e-02,
        #                     -2.5509e-02, -1.1217e-02, -2.5850e-02, -1.0414e-02, -3.5466e-02,
        #                     -1.9228e-02, 6.6134e-02, -8.6518e-03, -1.9620e-02, -1.2816e-02,
        #                     -2.2748e-02, 9.7364e-03, -8.6467e-03, -7.6724e-03, -1.5904e-02,
        #                     -2.1532e-02, -1.2969e-02, -7.0709e-03, -6.7640e-03, -1.5913e-02,
        #                     -8.1871e-03, -5.3848e-03, -1.9973e-02, -1.0092e-01, -2.7968e-02,
        #                     -1.9592e-02, -1.6870e-02, -1.7294e-02, -1.1857e-02, -1.8681e-02,
        #                     -3.1598e-02, -1.9135e-02, -9.3484e-02, -1.5351e-02, -2.2005e-02,
        #                     -6.3708e-03, -1.5302e-02, -2.6590e-02, -1.6007e-02, -3.1766e-02,
        #                     -2.2978e-02, -4.0257e-03, -1.6435e-03, -1.3427e-02, -1.7256e-02,
        #                     -4.7762e-04, -1.7345e-02, -1.9967e-02, -2.0178e-02, -7.5991e-03,
        #                     -8.6786e-03, -5.5343e-03, -2.3109e-02, -1.8284e-02, -9.5474e-04,
        #                     -1.0240e-03, -2.7650e-02, -1.1500e-02, -1.9704e-02, -1.2152e-02,
        #                     -5.5054e-03, -8.9523e-03, -1.5736e-02, 8.8114e-04, -2.0289e-02,
        #                     -4.8598e-03, -4.0729e-02, -1.7447e-02, -2.0635e-02, -1.9194e-02,
        #                     -1.4880e-03, -1.6642e-02, -1.1879e-02, -1.6033e-02, -3.0671e-02,
        #                     -1.6477e-02, -1.3104e-02, -6.3218e-04, -1.5913e-02, -1.6353e-02,
        #                     -1.0177e-02, -1.8300e-02, -1.9257e-02, -4.6872e-03, -1.9124e-02,
        #                     -1.6480e-02, -2.3118e-02, -3.0617e-03, -1.9778e-02, -1.0284e-02,
        #                     -3.5803e-03, -4.4318e-05, -1.4565e-02, -4.1273e-03, -2.1498e-02,
        #                     -1.7661e-02, -1.6043e-02, -2.6040e-02, -1.4441e-02, -9.9395e-03,
        #                     -9.0317e-03, -1.2087e-02, -2.0747e-02, -1.7486e-02, -1.1314e-02,
        #                     -2.5008e-02, 3.5960e-01, 4.6567e-03, -2.8229e-02, -1.4919e-02,
        #                     -1.8150e-02, -1.0866e-02, -2.6488e-02, -2.5328e-03, -2.4305e-02,
        #                     -1.8410e-02, -2.7910e-02, -1.3512e-02, -2.5245e-02, -2.2744e-02,
        #                     -1.5932e-02, -2.1175e-02, -1.4656e-02, -1.1476e-02, -2.1296e-02,
        #                     -2.2497e-02, 7.4823e-03, -1.5646e-02, -1.5749e-02, -1.2886e-01,
        #                     -1.1778e-02, -5.1732e-03, -2.7694e-02, -2.0384e-02, -1.8751e-02,
        #                     -1.1711e-02, -1.3219e-02, -1.7756e-02, -1.3359e-02, -2.7575e-03,
        #                     -6.2630e-03, -2.1703e-02, -1.5484e-02, -2.2764e-02, -1.6250e-02,
        #                     -1.5287e-02, 1.7300e-02, -1.8100e-02, -1.5910e-02, -2.0218e-02,
        #                     -1.3533e-02, -1.9378e-02, -5.1353e-03, -1.3445e-02, -1.0179e-02,
        #                     -1.5998e-02, -2.1023e-02, -8.9421e-03, -2.3921e-02, -7.9908e-03,
        #                     -2.2649e-02, -1.4676e-02, -4.3830e-03, -1.3992e-02, -1.8297e-02,
        #                     -1.0896e-02, -2.5316e-02, -1.1652e-02, -1.1272e-02, -9.1309e-03,
        #                     -1.8035e-02, -2.1521e-02, -1.7592e-02, -2.0304e-02, -1.1123e-02,
        #                     -2.0025e-02, -3.3089e-02, -7.5771e-03, -3.4157e-03, -1.1447e-02,
        #                     -1.4149e-02, -5.3260e-04]]])
        #
        # z = token
        # for i in range(b-1):
        #     z = torch.cat((z, token), dim=0)

        x = self.bneck(self.stem(x))
        for m in self.block:
            x, z = m(x, z)

        # 转成b个平铺一维向量
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        # 取第一个token
        z = z[:, 0, :].view(b, -1)
        # 最后一个维度拼接
        out = torch.cat((x, z), -1)

        out = self.head1(out)
        out = self.tanh(out)
        out = self.head2(out)
        return out
