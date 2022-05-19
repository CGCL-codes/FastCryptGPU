#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this file is jialeiguo and xiaoningwang's relu related coed which base on falcon

# from re import X
# from turtle import towards
# from xml.dom import xmlbuilder
# from xml.etree.ElementTree import XMLID
# from .mpc import mode, MPCTensor
# from .primitives.converters import convert, get_msb
# from .ptype import ptype as Ptype
# from dataclasses import dataclass
# from functools import wraps

# import CryptGPU
import os
import crypten
import torch
import numpy as np
import multiprocessing
# from scripts import multiprocess_launcher
# import scripts
# from crypten.mpc import prime_2_publicwrap3_sharing_generator

from crypten import communicator as comm
from crypten.common.tensor_types import is_tensor
from crypten.common.util import ConfigBase, pool_reshape, torch_cat, torch_stack
from crypten.cuda import CUDALongTensor

# from ..cryptensor import CrypTensor
# from ..encoder import FixedPointEncoder
# from .max_helper import _max_helper_all_tree_reductions
from crypten.mpc.primitives import resharing
from crypten.common.util import torch_stack
import time


def pc_replicate_shares(x_share):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    x_rep = torch.zeros_like(x_share.data)

    send_group = getattr(comm.get(), f"group{rank}{prev_rank}")
    recv_group = getattr(comm.get(), f"group{next_rank}{rank}")

    req1 = comm.get().isend(x_share.data, dst=prev_rank, group=send_group)
    req2 = comm.get().irecv(x_rep.data, src=next_rank, group=recv_group)

    req1.wait()
    req2.wait()

    if x_share.is_cuda:
        x_rep = CUDALongTensor(x_rep)

    return x_rep


# 质数域分享点积（a,b是与x类型、维度一致的tensor）
def funcDotProduct(a1byte, a2byte, b1byte, b2byte):
    if (isinstance(a1byte, CUDALongTensor)):
        a1byte = a1byte.tensor()
    if (isinstance(a2byte, CUDALongTensor)):
        a2byte = a2byte.tensor()
    if (isinstance(b1byte, CUDALongTensor)):
        b1byte = b1byte.tensor()
    if (isinstance(b2byte, CUDALongTensor)):
        b2byte = b2byte.tensor()

    rank = comm.get().get_rank()

    a1, a2, b1, b2 = a1byte, a2byte, b1byte, b2byte


    p11 = a1 * b1
    p12 = a1 * b2
    p13 = a2 * b1

    p1 = p11 + p12 + p13

    p1 = p1.byte()

    # world_size = comm.get().get_world_size()
    # prev_rank = (rank - 1) % world_size
    # next_rank = (rank + 1) % world_size

    # p2 = torch.zeros_like(p1.data)

    # send_group = getattr(comm.get(), f"group{rank}{prev_rank}")
    # recv_group = getattr(comm.get(), f"group{next_rank}{rank}")

    # req1 = comm.get().isend(p1.data, dst=prev_rank, group=send_group)
    # req2 = comm.get().irecv(p2.data, src=next_rank, group=recv_group)

    # req1.wait()
    # req2.wait()

    return p1


# 质数域分享异或（注意区分rank）(a,r是与r类型、维度一致的tensor)
# bit_r == 0： x[i]不变；bit_r == 1: x[i]取反，即 x[i] = 1 - x[i]
# 由于r在tensor里不能if，所以使用矩阵选择法，用对应位置为1的矩阵保留对应值，再将结果矩阵相加即可
def XORModPrime(a1, r0):
    if (isinstance(a1, CUDALongTensor)):
        a1 = a1.tensor()
    # if (isinstance(a2, CUDALongTensor)):
    #     a2 = a2.tensor()
    if (isinstance(r0, CUDALongTensor)):
        r0 = r0.tensor()
    r = r0.byte()
    rank = comm.get().get_rank()
    r0a1 = a1 + 0
    # r0a2 = a2 + 0
    if rank == 0:
        r1a1 = 257 - a1
        # r1a2 = 256 - a2
    else:
        r1a1 = 256 - a1 
        # r1a2 = 256 - a2

    tempfirst1 = (1 - r) * r0a1
    tempseond1 = r * r1a1
    # tempfirst2 = (1 - r) * r0a2
    # tempseond2 = r * r1a2

    ans1 = tempfirst1 + tempseond1
    # ans2 = tempfirst2 + tempseond2

    return ans1


# 连乘函数依赖函数，负责相邻两位的依次乘
# 没有改变tensor size，只是将有效数据存入前半区，主函数最后自行缩减
def funcMultiplyNeighbours(c1byte, c2byte, size):
    if (isinstance(c1byte, CUDALongTensor)):
        c1byte = c1byte.tensor()
    if (isinstance(c2byte, CUDALongTensor)):
        c2byte = c2byte.tensor()

    c1, c2 = c1byte.short(), c2byte.short()

    ans1 = torch.zeros_like(c1.data)

    for i in range(size // 2):
        ans1[i] = ((c1[2 * i] * c1[2 * i + 1]) + (c1[2 * i] * c2[2 * i + 1]) + 
                    (c2[2 * i] * c1[2 * i + 1]))
    ans1 = ans1.byte()

    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    ans2 = torch.zeros_like(ans1.data)

    send_group = getattr(comm.get(), f"group{rank}{prev_rank}")
    recv_group = getattr(comm.get(), f"group{next_rank}{rank}")

    req1 = comm.get().isend(ans1.data, dst=prev_rank, group=send_group)
    req2 = comm.get().irecv(ans2.data, src=next_rank, group=recv_group)

    req1.wait()
    req2.wait()

    return ans1, ans2


# 连乘函数，得到c[i]之后要判断其中有没有0，因此需要连乘
# 普通的乘法方法每个64位数连乘需要63次，即通信63次，复杂度太高，因此需要按位划分连乘
def funcCrunchMultiply(c1, c2):
    c1_2, c2_2 = funcMultiplyNeighbours(c1, c2, 64)
    c1_4, c2_4 = funcMultiplyNeighbours(c1_2, c2_2, 32)
    c1_8, c2_8 = funcMultiplyNeighbours(c1_4, c2_4, 16)
    c1_16, c2_16 = funcMultiplyNeighbours(c1_8, c2_8, 8)
    c1_32, c2_32 = funcMultiplyNeighbours(c1_16, c2_16, 4)
    c1_64, c2_64 = funcMultiplyNeighbours(c1_32, c2_32, 2)

    c1_64 = c1_64[0] + 0
    c2_64 = c2_64[0] + 0
    # c1_64 = c1_8[0]+0
    # c2_64 = c2_8[0]+0

    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    recvtensor = torch.zeros_like(c1_64.data)

    send_group = getattr(comm.get(), f"group{rank}{prev_rank}")
    recv_group = getattr(comm.get(), f"group{next_rank}{rank}")

    req1 = comm.get().isend(c2_64.data, dst=prev_rank, group=send_group)
    req2 = comm.get().irecv(recvtensor.data, src=next_rank, group=recv_group)

    req1.wait()
    req2.wait()

    localtensor = c1_64 + c2_64
    reconst = localtensor + recvtensor

    onetensor = torch.ones_like(reconst.data)
    # 合并两个tensor， reconst == 0的地方保留，再0变1，1变0
    tbeta = torch.where(reconst == 0, reconst, onetensor)
    tbeta = tbeta - torch.ones_like(reconst.data)
    tbeta = torch.where(tbeta == 0, tbeta, onetensor)

    return tbeta


# 二进制分享异或
def XORModBinary(tbetabinary, betabinary1):
    rank = comm.get().get_rank()
    pc1 = betabinary1 + 0
    # pc2 = betabinary2 + 0

    if rank == 0:
        pc1 = (betabinary1 + tbetabinary) & 1
    # elif rank == 2:
    #     pc2 = (betabinary2 + tbetabinary) & 1

    return pc1

# def timer_privatecompare(func):
#     """用来计算理论中属于预计算的时间"""

#     def wrapper_timer(*args, **kwargs):
#         start_time = time.perf_counter()  # 1
#         value = func(*args, **kwargs)
#         end_time = time.perf_counter()  # 2
#         run_time = end_time - start_time  # 3
#         comm.get().time_privatecompare += run_time
#         return value

#     return wrapper_timer
# 隐私比较（参数x是按位分享的tensor， r是公有的tensor, beta是随机bit的tensor,格式同x）
# 返回pc结果的分享
# @timer_privatecompare
def PrivateCompare(x, r, beta1, betabinary1):

    rank = comm.get().get_rank()
    # x2 = pc_replicate_shares(x)
    # beta2 = pc_replicate_shares(beta1)
    # betabinary2 = pc_replicate_shares(betabinary1)

    # if rank == 0 :
    #   print(f"x1 = {x}x2 = {x2}beta1 = {beta1}beta2 = {beta2}betab1 = {betabinary1}betab2 = {betabinary2}r = {r}")
    # 求2beta - 1(twoBetaMinusOne)
    # 获取beta1/beta2（目前是按照x的格式传参进来，后续可能改进）
    twoBetaMinusOne1 = 2*beta1
    # twoBetaMinusOne2 = 2*beta2
    if rank==0:
        twoBetaMinusOne1 -= 1
    # elif rank==2:
        # twoBetaMinusOne2 -= 1
    twoBetaMinusOne2 = pc_replicate_shares(twoBetaMinusOne1)

    # 求x[i] - r[i](diff)
    # 防止溢出，减法前先加67
    diff1 = x + 0
    # diff2 = x2 + 0
    bit_r = torch.stack(
        [((r >> i) & 1).byte() for i in range(l)], 
    dim=0, out=None)
    if rank == 0:
        diff1 = diff1 - bit_r
    # if rank == 2:
    #     diff2 = diff2 - bit_r
    diff2 = pc_replicate_shares(diff1)
    # 求u[i] = (-1)^beta * (x[i] - r[i]) (XMinusR)
    # 秘密分享点积（funcDotProduct）(已完成)

    XMinusR1 = funcDotProduct(diff1, diff2, twoBetaMinusOne1, twoBetaMinusOne2)

    # 求w[i] = x[i] XOR r[i] (tempN) 和 c[i] (c)
    # 秘密分享异或（XORModPrime）(注意区分rank)（已完成）
    # tempN1 = torch.zeros_like(x.data)
    # tempN2 = torch.zeros_like(x.data)
    # suma是少一维的0 tensor，与r同维度，生成方法待验证
    suma1 = torch.zeros_like(r.data)
    # suma2 = torch.zeros_like(r.data)
    c1 = torch.zeros_like(x.data)
    # c2 = torch.zeros_like(x.data)
    range64 = range(l)
    tempN1 = XORModPrime(x, bit_r)
    for i in reversed(range64):
        c1[i] = suma1 + 0
        # c2[i] = suma2 + 0
        # bit_r = (r >> i) & 1
        # print(f"rank = {rank}i = {i}x1[i] = {x[i]}bit_r = {bit_r}\n")
        # tempN1, tempN2 = XORModPrime(x[i], x2[i], bit_r[i])
        # mpc_out = comm.get().all_reduce(tempN1)
        # true = comm.get().all_reduce(x[i])
        # if rank == 0:
        #     print(f"x = {true} r = {bit_r} out = {mpc_out}\n")
        suma1 = suma1 + tempN1[i]
        # suma2 = suma2 + tempN2

    # print(f"rank = {rank}c1 = {c1}\n")

    c1 = c1 + XMinusR1
    # c2 = c2 + XMinusR2
    if rank == 0:
        c1 = c1 + 1
    # if rank == 2:
    #     c2 = c2 + 1


    c1 = comm.get().all_reduce(c1)
    # c1 = c1.
    # c1 = c1-1
    # c1 = c1 >> 7
    # c1 = 1 - c1
    onetensor = torch.ones_like(c1.data)
    c1 = torch.where(c1 == 0, c1, onetensor)
    # print(f'c1={c1}')
    tbetabinary = torch.ones_like(r,dtype=torch.uint8,device=r.device)
    for i in range(l):
        tbetabinary = tbetabinary*c1[i]
    tbetabinary = 1 - tbetabinary
    
    # print(f'tbetabinary={tbetabinary}')
        
        
    # 计算beta’(即计算d = m*c[0]*c[1]......是否为0）(二进制分享)
    # 连乘函数（funcCrunchMultiply） （完成）

    # 计算结果
    # 二进制分享beta，二进制分享异或(XORModBinary)(已完成)

    pc1 = XORModBinary(tbetabinary, betabinary1)


    if (isinstance(pc1, CUDALongTensor)):
        pc1 = pc1.tensor()
    # if (isinstance(pc2, CUDALongTensor)):
    #     pc2 = pc2.tensor()
    pc1 = pc1.byte()
    # pc2 = pc2.byte()

    return pc1




# 测试
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
size = [2]
size_bit = size.copy()
size_bit.insert(0, 64)
device = 'cuda'
dtype = torch.uint8
l = 64



# 质数域点积测试完成
def producttest():
    rank = comm.get().get_rank()
    a1, a2, a3, b1, b2, b3 = torch.tensor([33], dtype=dtype, device=device), torch.tensor([34], dtype=dtype,
                                                                                          device=device), torch.tensor(
        [1], dtype=dtype, device=device), torch.tensor([33], dtype=dtype, device=device), torch.tensor([33],
                                                                                                       dtype=dtype,
                                                                                                       device=device), torch.tensor(
        [0], dtype=dtype, device=device)
    if rank == 0:
        p1, p2 = funcDotProduct(a1, a2, b1, b2)
        print(f"rank0 p1 = {p1}\n")
        print(f"rank0 p2 = {p2}\n")
    if rank == 1:
        p1, p2 = funcDotProduct(a2, a3, b2, b3)
        print(f"rank1 p1 = {p1}\n")
        print(f"rank1 p2 = {p2}\n")
    if rank == 2:
        p1, p2 = funcDotProduct(a3, a1, b3, b1)
        print(f"rank2 p1 = {p1}\n")
        print(f"rank2 p2 = {p2}\n")


# 质数域异或测试完成
def XORPrimetest():
    rank = comm.get().get_rank()
    a1, a2, a3, r = torch.tensor([1], dtype=dtype, device=device), torch.tensor([32], dtype=dtype,
                                                                                device=device), torch.tensor([34],
                                                                                                             dtype=dtype,
                                                                                                             device=device), torch.tensor(
        [0], dtype=dtype, device=device)
    if rank == 0:
        p1, p2 = XORModPrime(a1, a2, r)
        print(f"rank0 p1 = {p1}\n")
        print(f"rank0 p2 = {p2}\n")
    if rank == 1:
        p1, p2 = XORModPrime(a2, a3, r)
        print(f"rank1 p1 = {p1}\n")
        print(f"rank1 p2 = {p2}\n")
    if rank == 2:
        p1, p2 = XORModPrime(a3, a1, r)
        print(f"rank2 p1 = {p1}\n")
        print(f"rank2 p2 = {p2}\n")


# 二进制分享异或测试完成
def XORBinarytest():
    rank = comm.get().get_rank()
    tb, b1, b2, b3 = torch.tensor([1], dtype=dtype, device=device), torch.tensor([1], dtype=dtype,
                                                                                 device=device), torch.tensor([1],
                                                                                                              dtype=dtype,
                                                                                                              device=device), torch.tensor(
        [1], dtype=dtype, device=device)

    if rank == 0:
        p1, p2 = XORModBinary(tb, b1, b2)
        print(f"rank0 p1 = {p1}\n")
        print(f"rank0 p2 = {p2}\n")
    if rank == 1:
        p1, p2 = XORModBinary(tb, b2, b3)
        print(f"rank1 p1 = {p1}\n")
        print(f"rank1 p2 = {p2}\n")
    if rank == 2:
        p1, p2 = XORModBinary(tb, b3, b1)
        print(f"rank2 p1 = {p1}\n")
        print(f"rank2 p2 = {p2}\n")


# 连乘测试完成
def CrunchMultiplytest():
    rank = comm.get().get_rank()
    c1 = torch.tensor([[0], [5], [15], [5], [1], [5], [15], [5]], dtype=dtype, device=device)
    c2 = torch.tensor([[33], [6], [21], [66], [33], [6], [21], [66]], dtype=dtype, device=device)
    c3 = torch.tensor([[33], [30], [14], [1], [33], [30], [14], [1]], dtype=dtype, device=device)

    if rank == 0:
        tbeta = funcCrunchMultiply(c1, c2)
        print(f"rank0 tbeta = {tbeta}\n")
    if rank == 1:
        tbeta = funcCrunchMultiply(c2, c3)
        print(f"rank1 tbeta = {tbeta}\n")
    if rank == 2:
        tbeta = funcCrunchMultiply(c3, c1)
        print(f"rank2 tbeta = {tbeta}\n")


# pc整体测试完成
def pctest():
    rank = comm.get().get_rank()
    x1 = torch.tensor([[0], [5], [15], [0], [0], [5], [16], [0]], dtype=dtype, device=device)
    x2 = torch.tensor([[33], [30], [15], [0], [33], [30], [15], [0]], dtype=dtype, device=device)
    x3 = torch.tensor([[34], [32], [37], [0], [34], [32], [37], [0]], dtype=dtype, device=device)
    r = torch.tensor([1], dtype=dtype, device=device)
    beta1 = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0]], dtype=dtype, device=device)
    beta2 = torch.tensor([[33], [33], [33], [33], [33], [33], [33], [33]], dtype=dtype, device=device)
    beta3 = torch.tensor([[34], [34], [34], [34], [34], [34], [34], [34]], dtype=dtype, device=device)
    betab1 = torch.tensor([0], dtype=dtype, device=device)
    betab2 = torch.tensor([0], dtype=dtype, device=device)
    betab3 = torch.tensor([0], dtype=dtype, device=device)

    if rank == 0:
        p1, p2 = PrivateCompare(x1, r, beta1, betab1)
        print(f"rank0 p1 = {p1}\n")
        print(f"rank0 p2 = {p2}\n")
    if rank == 1:
        p1, p2 = PrivateCompare(x2, r, beta2, betab2)
        print(f"rank1 p1 = {p1}\n")
        print(f"rank1 p2 = {p2}\n")
    if rank == 2:
        p1, p2 = PrivateCompare(x3, r, beta3, betab3)
        print(f"rank2 p1 = {p1}\n")
        print(f"rank2 p2 = {p2}\n")


# def debug_pc():
    # global device
    # device = 'cpu'
    # print(f"a1 = {a1}\na2 = {a2}\na3 ={a3}\nb1 = {b1}\nb2 = {b2}\nb3 = {b3}")
    # launcher = multiprocess_launcher.MultiProcessLauncher(
    #     3, pctest,
    # )
    # launcher.start()
    # launcher.join()
    # launcher.terminate()


# if __name__ == '__main__':
    # x1 = torch.empty(size = size_bit, dtype = torch.uint8, device = device).random_(0, to = 67)
    # x2 = torch.empty(size = size_bit, dtype = torch.uint8, device = device).random_(0, to = 67)
    # x1, x2 = x1.short(), x2.short()
    # x = (x1*x2)%67
    # x = x.byte()
    # x1 = torch.tensor([[1],[5],[15],[0],[0],[5],[15],[0]], dtype = dtype, device = device)
    # print(f"{x1[0]}")
    # print(f"x1 = {x1} x2 = {x2}")
    # print(f"x1*x2 = {x}")

    # debug_pc()