#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this file is jialeiguo and xiaoningwang's relu related coed which base on falcon
import multiprocessing

import crypten
from crypten.common.rng import generate_kbit_random_tensor
from crypten.common.util import torch_stack
from crypten.encoder import FixedPointEncoder
from crypten.mpc import MPCTensor
from scripts import multiprocess_launcher

import time
import torch
from crypten import communicator as comm
from crypten.cuda import CUDALongTensor
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
from crypten.mpc.private_compare_w import PrivateCompare
from crypten.mpc.prime_2_publicwrap3_sharing_generator import generate_random_ring_element_bits_share_wrap3,wrap3_local,wrap2_local,generate_2_p_share,generate_2_l_share

l = 64
""" 
    根据falcon的3.4算法所写
    输入a 应当是torch.tensor,之后计算都按照此格式
    返回的结果是2进制的torch.tensor uint8
"""
def wrap3_mpc(a):
    rank = comm.get().get_rank()
    #1 获取公共随机数
    x_l,x_bit_p,alpha = generate_random_ring_element_bits_share_wrap3(a.size(),device=a.device)
    #2 r_i = a + x mod L
    r_i = a + x_l
    #3 beta = wrap2_local(a,x,l)
    beta,useless = wrap2_local(a, x_l)

    #4 公开r的值
    r_prev,r_next = reconstruct(r_i)
    r = r_i+r_prev+r_next
    # r = comm.get().all_reduce(r_i)

    #检验正确性
    # x_sum = comm.get().all_reduce(x_l)
    # r_sum = comm.get().all_reduce(r_i)
    # a_sum = comm.get().all_reduce(a)
    # if rank == 0:
    #     print(f'公开值r的获取是否成功={(a_sum+x_sum).equal(r_sum)}')

    #5 公开计算r1+r2+r3是否绕环
    delta = wrap3_local(r_i,r_prev,r_next)
    #6
    pc_beta,pc_beta_p = generate_2_p_share(x_l.size(),x_l.device)
    eta = PrivateCompare(x_bit_p,r+1,pc_beta_p,pc_beta)
    # eta = eta[0]
    # if rank == 0:
    #     eta = eta^1
    # eta = private_compare(x_l,r+1)

    #7 每方计算最终theta应有结果
    if rank == 0:
        theta = (beta^alpha^eta^delta)
    elif rank ==1:
        theta = (beta^alpha^eta)
    else:
        theta = (beta^alpha^eta)
    return theta

"""每方获得解密后的r=r1+r2+r3,ri就是torch.tensor """
def reconstruct(r_i):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    r_prev = torch.zeros_like(r_i.data)
    r_next = torch.zeros_like(r_i.data)

    send_group2prev = getattr(comm.get(), f"group{rank}{prev_rank}")
    send_group2next = getattr(comm.get(), f"group{rank}{next_rank}")
    recv_group5prev = getattr(comm.get(), f"group{prev_rank}{rank}")
    recv_group5next = getattr(comm.get(), f"group{next_rank}{rank}")
    #注意收发顺序，避免死锁
    req1 = comm.get().isend(r_i.data, dst=prev_rank, group=send_group2prev)
    req2 = comm.get().irecv(r_next.data, src=next_rank, group=recv_group5next)

    req3 = comm.get().isend(r_i.data, dst=next_rank, group=send_group2next)
    req4 = comm.get().irecv(r_prev.data, src=prev_rank, group=recv_group5prev)

    req1.wait()
    req2.wait()
    req3.wait()
    req4.wait()

    return r_prev,r_next

"""获取最高符号位
    输入为ArithmeticTensor
    输出为2进制分享，结果暂定为torch.tensor uint8
"""
def gw_get_msb(input):
    a = input
    if isinstance(a, ArithmeticSharedTensor):
        a = a._tensor
    if isinstance(a,CUDALongTensor):
        a = a.tensor()
    theta = wrap3_mpc(a<<1)
    sign_bit = theta^((a>>(l-1)).byte())
    return sign_bit


"""
基于cryptgpu的三方不经意传输实现混合相乘
x 为ArithmeticTensor
y_2 为torch.tensor uint8 
返回MPCTensor
"""
def mixed_mul_cryptgpu(x, y_2, roles=[0, 1, 2]):
    y_2 = y_2.long()
    sender, receiver, helper = roles
    groups2r = getattr(comm.get(), f"group{sender}{receiver}")
    groups2h = getattr(comm.get(), f"group{sender}{helper}")
    groupr2h = getattr(comm.get(), f"group{receiver}{helper}")

    zero_share = ArithmeticSharedTensor.PRZS(y_2.size(), device=y_2.device).share


    rank = x.rank
    if rank == sender:
        xs = torch.ones_like(y_2.data)
        b1, b3 = y_2, replicate_shares(y_2)
        if isinstance(b3,CUDALongTensor):
            b3 = b3.tensor()

        b1, b3 = b1 & 1, b3 & 1

        if sender > helper:
            b1, b3 = b3, b1
        # r 是为了让receiver无法重构信息
        ring_size = 2 ** 64
        r = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)
        m0 = (b1 ^ b3 ^ 0) * xs - r
        m1 = (b1 ^ b3 ^ 1) * xs - r

        w0 = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)
        w1 = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)

        req0 = comm.get().isend(torch_stack([m0 ^ w0, m1 ^ w1]), dst=receiver, group=groups2r)
        req1 = comm.get().isend(torch_stack([w0, w1]), dst=helper, group=groups2h)
        req0.wait()
        req1.wait()

        a = ArithmeticSharedTensor.from_shares(zero_share + r, src=rank)

    if rank == receiver:
        b2, b1 = y_2, replicate_shares(y_2)
        if isinstance(b1,CUDALongTensor):
            b1 = b1.tensor()

        if sender > helper:
            b1, b2 = b2, b1

        m_b = torch.zeros_like(torch_stack([b1, b2])).data
        w_b2 = torch.zeros_like(b2).data

        req0 = comm.get().irecv(m_b, src=sender, group=groups2r)
        req1 = comm.get().irecv(w_b2, src=helper, group=groupr2h)
        req0.wait()
        req1.wait()

        size = b1.size()
        bin_bits = b2.flatten().data
        m_b = m_b.view(2, -1)
        m_b2 = m_b[bin_bits, torch.arange(bin_bits.size(0))]
        m_b2 = m_b2.view(size)
        message = m_b2 ^ w_b2

        a = ArithmeticSharedTensor.from_shares(zero_share + message, src=rank)

    if rank == helper:
        b3, b2 = y_2, replicate_shares(y_2)
        if isinstance(b2,CUDALongTensor):
            b2 = b2.tensor()

        if sender > helper:
            b3, b2 = b2, b3

        w = torch.zeros_like(torch_stack([b3, b3])).data
        req0 = comm.get().irecv(w, src=sender, group=groups2h)
        req0.wait()

        size = b3.size()
        bin_bits = b2.flatten().data

        w = w.view(2, -1)
        w_b2 = w[bin_bits, torch.arange(bin_bits.size(0))]
        w_b2 = w_b2.view(size)

        req1 = comm.get().isend(w_b2, dst=receiver, group=groupr2h)
        req1.wait()

        a = ArithmeticSharedTensor.from_shares(zero_share, src=rank)

    a.encoder = x.encoder

    a.encoder = FixedPointEncoder(precision_bits=None)
    scale = a.encoder._scale
    a *= scale

    return MPCTensor.from_shares((x * a).share)


"""RELU
    输入为MPCSharedTensor
    输出为relu后的MPCSharedTensor
"""
def gw_relu(x):
    rank = comm.get().get_rank()
    sign_bit = gw_get_msb(x._tensor)
    # print(comm.get().all_reduce(sign_bit)&1)
    if rank == 0:
        sign_bit = sign_bit^1
    return select_share(x._tensor, sign_bit)
    
    # return mixed_mul_aby3(x._tensor, sign_bit)


""" 暂时模拟Fpc效果
    实际上比较的是绝对值的大小，对应符号数要做额外处理
"""
def private_compare(x_l,r):
    rank = comm.get().get_rank()
    x = comm.get().all_reduce(x_l)
    ones = torch.ones_like(x.data)
    zeros = torch.zeros_like(x.data).byte()

    # 符号位的提取,为1为负
    x_msb = (x >> (l - 1)).byte()
    r_msb = (r >> (l - 1)).byte()



    # 为1为有效的元素
    p2 = 1-(x_msb|r_msb)
    n2 = x_msb&r_msb
    #为0不管
    xp_rn = (1-x_msb)&r_msb
    xn_rp = x_msb&(1-r_msb)


    with_sign = torch.where(x < r, zeros, p2+n2)
    with_sign = with_sign.byte()

    out = xn_rp + with_sign

    if rank == 0:
        #print(p2+n2+xp_rn+xn_rp)
        return out.byte()
    else:
        return zeros.byte()


def replicate_shares(x_share):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    x_rep = torch.zeros_like(x_share.data)

    send_group = getattr(comm.get(), f"group{rank}{next_rank}")
    recv_group = getattr(comm.get(), f"group{prev_rank}{rank}")

    tic = time.perf_counter()
    req1 = comm.get().isend(x_share.data, dst=next_rank, group=send_group)
    req2 = comm.get().irecv(x_rep.data, src=prev_rank, group=recv_group)

    req1.wait()
    req2.wait()

    toc = time.perf_counter()

    comm.get().comm_time += toc - tic

    return x_rep


"""
基于aby3的三方不经意传输实现混合相乘
x 为ArithmeticTensor
y_2 为torch.tensor uint8 
返回MPCTensor
"""
def mixed_mul_aby3(x, y_2):
    
    y_2 = y_2.long()
    group01 = getattr(comm.get(), f"group{0}{1}")
    group02 = getattr(comm.get(), f"group{0}{2}")
    group12 = getattr(comm.get(), f"group{1}{2}")

    zero_share = ArithmeticSharedTensor.PRZS(y_2.size(), device=y_2.device).share

    #为了一次传播，进行打包
    rank = x.rank
    x = x.share
    if isinstance(x,CUDALongTensor):
        x = x.tensor()
    pre_x_y_2 = torch.stack([x,y_2], dim=0, out=None)

    #接下来x的秘密分享下标为对应rank+1，这样便于按照设计算法实现代码
    #p1担任发送方和接收方
    if rank == 0:
        
        b1, tmp = y_2, replicate_shares(pre_x_y_2)
        b3 = tmp[1]
        x1, x3 = x, tmp[0]

        m_b = torch.zeros_like(torch.stack([b1, b3], dim=0, out=None)).data
        w_b1 = torch.zeros_like(b3).data
        # r 是为了让receiver无法重构信息
        ring_size = 2 ** 64
        r = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)

        m0 = ((b1 ^ b3 ^ 0) * x1) - r
        m1 = ((b1 ^ b3 ^ 1) * x1) - r

        w0 = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)
        w1 = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)

        req1 = comm.get().isend(torch.stack([w0, w1], dim=0, out=None), dst=2, group=group02)
        req0 = comm.get().isend(torch.stack([m0 ^ w0, m1 ^ w1], dim=0, out=None), dst=1, group=group01)
        req0.wait()
        req1.wait()

        #p1担任接收方
        req2 = comm.get().irecv(m_b, src=2, group=group02)
        req3 = comm.get().irecv(w_b1, src=1, group=group01)
        req2.wait()
        req3.wait()

        size = b1.size()
        bin_bits = b1.flatten().data
        m_b = m_b.view(2, -1)
        m_b1 = m_b[bin_bits, torch.arange(bin_bits.size(0))]
        m_b1 = m_b1.view(size)
        message = m_b1 ^ w_b1
        #p1担任接收方
        # print(f'b1={b1} rank 0 recieve message={message}')
        # print(f'rank 1 should recieve message={torch_stack([m0, m1])}')
        # print(f'rank 1 should recieve m_b2={torch_stack([m0 ^ w0, m1 ^ w1])}')
        a = zero_share + r + message

    #p2担任接收方和辅助方
    if rank == 1:
        b2, tmp = y_2, replicate_shares(pre_x_y_2)
        b1 = tmp[1]
        x2, x1 = x, tmp[0]

        m_b = torch.zeros_like(torch.stack([b1, b2], dim=0, out=None)).data
        req0 = comm.get().irecv(m_b, src=0, group=group01)
        

        #p2担任辅助方
        tmp = torch.zeros_like(torch.stack([b1, b2, b2], dim=0, out=None)).data
        req2 = comm.get().irecv(tmp, src=2, group=group12)
        req2.wait()
        req0.wait()
        w = tmp[0:2]
        w_b2 = tmp[2]
        
        size = b1.size()
        bin_bits = b1.flatten().data

        w = w.view(2, -1)
        w_b1 = w[bin_bits, torch.arange(bin_bits.size(0))]
        w_b1 = w_b1.view(size)

        req3 = comm.get().isend(w_b1, dst=0, group=group01)
        
        #p2担任辅助方

        

        size = b2.size()
        bin_bits = b2.flatten().data
        m_b = m_b.view(2, -1)
        m_b2 = m_b[bin_bits, torch.arange(bin_bits.size(0))]
        m_b2 = m_b2.view(size)
        message = m_b2 ^ w_b2
        req3.wait()
        # print(f'rank 1 recieve m_b2={m_b2}')
        # print(f'rank 1 recieve w_b2={w_b2}')
        # print(f'b2={b2} rank 1 recieve message={message}')
        a = zero_share + message

    #p3担任辅助方和发送方
    if rank == 2:
        b3, tmp = y_2, replicate_shares(pre_x_y_2)
        b2 = tmp[1]
        x3, x2 = x, tmp[0]

        w = torch.zeros_like(torch.stack([b3, b3], dim=0, out=None)).data
        req2 = comm.get().irecv(w, src=0, group=group02)
        req2.wait()
        size = b2.size()
        bin_bits = b2.flatten().data

        w = w.view(2, -1)
        w_b2 = w[bin_bits, torch.arange(bin_bits.size(0))]
        w_b2 = w_b2.view(size)
        #p3担任发送方
        ring_size = 2 ** 64
        r = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)

        m0 = ((b2 ^ b3 ^ 0) * (x2+x3)) - r
        m1 = ((b2 ^ b3 ^ 1) * (x2+x3)) - r
        w0 = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)
        w1 = torch.empty(size=y_2.size(), dtype=torch.long, device=y_2.device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)

        req0 = comm.get().isend(torch.stack([w0, w1, w_b2], dim=0, out=None), dst=1, group=group12)
        req1 = comm.get().isend(torch.stack([m0 ^ w0, m1 ^ w1], dim=0, out=None), dst=0, group=group02)
        req0.wait()
        
        #p3担任发送方
    
        

        req1.wait()
        # print(f'rank 0 should recieve message={torch_stack([m0, m1])}')
        # print(f'rank 1 should recieve w_b2={w_b2}')
        a = zero_share + r

    return MPCTensor.from_shares(a)


"""
基于falcon的ss实现混合相乘
x 为ArithmeticTensor
y_2 为torch.tensor uint8 
返回MPCTensor
"""
def select_share(x, y_2):
    y_2 = y_2.long()

    #为了一次传播，进行打包
    rank = x.rank
    c_2,c_l = generate_2_l_share(x.size(),x.device)
    # print(f'c_2={comm.get().all_reduce(c_2)&1}')
    # print(f'c_l={comm.get().all_reduce(c_l)}')
    # c_l = MPCTensor.from_shares(c_l)._tensor
    e_2 = c_2^y_2
    e = comm.get().all_reduce(e_2)&1
    
    if rank == 0:
        neg_c_l_plus1 = 1-c_l
    else:
        neg_c_l_plus1 = -c_l
    d = neg_c_l_plus1*e
    d += c_l*(1-e)
    d = ArithmeticSharedTensor.from_shares(d,src=rank)
    d.encoder = x.encoder

    d.encoder = FixedPointEncoder(precision_bits=None)
    scale = d.encoder._scale
    d *= scale

    tmp = ArithmeticSharedTensor.PRZS(x.size(),device=x.device)
    tmp.encoder = FixedPointEncoder(precision_bits=None)

    z = d*(x-tmp)+tmp

    return MPCTensor.from_shares(z.share)





""" 测试正确性的部分 """
# dtype = torch.long
# device = 'cpu'
#
#
# def debug_reconstruct():
#     rank = comm.get().get_rank()
#     r = torch.tensor([rank])
#     r1,r2 = reconstruct(r)
#     print(f'{r1}{r2}')
#
#
# def debug_fake_pc():
#     rank = comm.get().get_rank()
#     x = {0: torch.tensor([2, 13,10,-10], dtype=dtype, device=device),
#          1: torch.tensor([-1, -10,14,9], dtype=dtype, device=device),
#          2: torch.tensor([2, 2,-30,-1], dtype=dtype, device=device)}
#     r = torch.tensor([1, -300,100,-10], dtype=dtype, device=device)
#     tmp = private_compare(x[rank], r)
#     if rank == 0:
#         print(tmp)
#
#
# def debug_true_pc():
#     dtype = torch.long
#     device = 'cpu'
#     rank = comm.get().get_rank()
#     # x = {0: torch.tensor([2, 13,10,-10], dtype=dtype, device=device),
#     #      1: torch.tensor([-1, -10,14,9], dtype=dtype, device=device),
#     #      2: torch.tensor([2, 2,-30,-1], dtype=dtype, device=device)}
#     # r = torch.tensor([1, 6,1,-10], dtype=dtype, device=device)
#     x_true = torch.tensor([-2,-1,4,6666], dtype=dtype, device=device)#-2,-1,4,6666
#     r = torch.tensor([-1,10,2,7777], dtype=dtype, device=device)#-1,10,2,7777
#     # x_true = x[0]+x[1]+x[2]
#     x_i_p = torch_stack([((x_true >> i) & 1).byte() for i in range(l)])
#
#     zeros = torch.zeros_like(x_i_p)
#     pc_beta, pc_beta_p = generate_2_p_share(x_true.size(), x_true.device)
#
#
#     if(rank==0):
#         tmp = PrivateCompare(x_i_p, r,pc_beta_p,pc_beta)
#     else:
#         tmp = PrivateCompare(zeros, r,pc_beta_p,pc_beta)
#     tmp = tmp
#     tmp = (comm.get().all_reduce(tmp) & 1)
#     if rank == 0:
#         # print(f'x_i_p={x_i_p}')
#         print(f'x>=r={x_true}>{r}={tmp}')
        
#
#
# def debug_wrap3_mpc():
#     rank = comm.get().get_rank()
#     x = {0:torch.tensor([-1, 13, 7, -3], dtype=dtype, device=device),1:torch.tensor([-1, -10, 3, 2], dtype=dtype, device=device),2:torch.tensor([-2, 2, -1, -3], dtype=dtype, device=device)}
#
#     wrap01, res = wrap2_local(x[0], x[1])
#     wrap_res, tmp = wrap2_local(res, x[2])
#     if rank ==0:
#         print(f'wrpa01={wrap01}')
#         print(f'wrpa_res={wrap_res}')
#     t1=comm.get().all_reduce(wrap3_mpc(x[rank]))&1
#
#     if rank==0:
#         print(f'wrpa3={t1}')
#         print(f'wrap3_mpc==wrap3_local 结果为 {t1.equal(wrap3_local(x[0],x[1],x[2]))}')
#
#
# def debug_msb():
#     rank = comm.get().get_rank()
#     x_true = torch.LongTensor([0,2,1<<40-1,-1,-2,-1<<30])
#     x = crypten.cryptensor(x_true, device="cpu", requires_grad=False)
#     sign_bit=gw_get_msb(x._tensor)
#     sign_bit = comm.get().all_reduce(sign_bit)&1
#     if(rank == 0):
#         print(x_true)
#         print(sign_bit)
# #
# #
# def debug_mix_mul():
#     dtype = torch.long
#     rank = comm.get().get_rank()
#     x_true = torch.tensor([667,10,3,-1,-99,-2],dtype=dtype,device='cuda')
#     x = crypten.cryptensor(x_true, device=x_true.device, requires_grad=False)
#     sign_bit = torch.tensor([0,1,1,0,1,1],dtype=torch.uint8,device='cuda')
#
#     sign_bit = comm.get().all_reduce(sign_bit)&1
#     if rank == 0:
#         print(sign_bit)
#     # print(x.get_plain_text(0))
#     # if rank == 0:
#     #     sign_bit = sign_bit^1
#
#     after_relu = select_share(x._tensor,sign_bit)
#     # after_relu = mixed_mul_aby3(x._tensor,sign_bit)
#     decrypt = after_relu.get_plain_text(0)
#     if rank == 0:
#         print(decrypt)
#
#
# def debug_gw_relu():
#     dtype = torch.long
#     rank = comm.get().get_rank()
#     x_true = torch.tensor([61,52,589,0,-(166),-666],dtype=dtype,device='cpu')
#     x = crypten.cryptensor(x_true, device=x_true.device, requires_grad=False)
#     after_relu = gw_relu(x)
#     decrypt = after_relu.get_plain_text(0)
#     if rank == 0:
#         print(decrypt)
#         print(decrypt.dtype)
#
#
# def learn():
#     a = torch.tensor([256,255,-1],dtype=torch.uint8)
#     b = torch.tensor([2,1,-1],dtype=torch.uint8)
#     print(a+b)
#
# def mixed_mul_compare():
#     rank = comm.get().get_rank()
#     size = 100
#     sign = torch.tensor([i for i in range(size)], dtype=torch.uint8, device=device)
#     x_true = torch.tensor([i for i in range(size)],dtype=dtype,device=device)
#     x = crypten.cryptensor(x_true, device=x_true.device, requires_grad=False)
#
#     start_time = time.perf_counter()
#     after_relu0 = select_share(x._tensor,sign_bit)
#     end_time = time.perf_counter()
#     print(f'select_share 花费时间{end_dim-start_time}')
#
#     start_time = time.perf_counter()
#     after_relu1 = mixed_mul_aby3(x._tensor,sign_bit)
#
#     end_time = time.perf_counter()
#     print(f'mixed_mul_aby3 花费时间{end_dim-start_time}')
#
#     decrypt0 = after_relu0.get_plain_text(0)
#     decrypt1 = after_relu1.get_plain_text(0)
#     if rank == 0:
#         print(decrypt0)
#         print(decrypt1)
#
#
# if __name__ == '__main__':
#     launcher = multiprocess_launcher.MultiProcessLauncher(
#         3, debug_gw_relu
#     )
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
#     # learn()