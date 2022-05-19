#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this file is jialeiguo and xiaoningwang's relu related coed which base on falcon

import torch
import os
import multiprocessing
from multiprocessing import Manager
import crypten.communicator as comm
from crypten.cuda import CUDALongTensor
from crypten.mpc.primitives import ArithmeticSharedTensor
# from scripts import multiprocess_launcher
from crypten.common.util import torch_stack
import time

#
#默认在cryptgpu的64位整数域下做
l = 64
#用于&来实现减符号位
num_and = 0
for i in range(l-1):
    num_and += 1<<i
#质数域默认是67
p = 67
SENTINEL = -1
dtype = torch.long


def timer_precomq(func):
    """用来计算理论中属于预计算的时间"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        old_comm_bytes = comm.get().comm_bytes
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        comm.get().time_precomq += run_time
        comm.get().comm_bytes = old_comm_bytes
        return value

    return wrapper_timer
""" 
    根据输入的size，产生一个对应size的随机tensor的64位的秘密分享,该数自带精度，和其在p=256质数域下的bit秘密分享,以及wrap3(x1,x2,x3,l)的结果,结果按序返回;
    1 没有严格的遵守安全规范：要用第四方服务器来产生这个随机值，而是和crypten一样，在第一方本地生成
    2 bit的分享就是dtype=torch.uint8' torch.tensor or CUDALongTensor 由torch_stack导入,其计算模仿circuit自己写一个秘密分享下的计算——》要做乘法，自己整一个
    3 bit的分享在cuda情况下要单独处理:自己整个CUDAShortTensor-》dtype=torch.uint8可以无损嵌入gpu
    4 2进制的分享就是dtype=torch.uint8' torch.tensor:为1就是对应位置发生绕环
    5 L的分享就是dtype=torch.int64' torch.tensor
"""
@timer_precomq
def generate_random_ring_element_bits_share_wrap3(size, ring_size=(2 ** l), device='cuda'):
    global dtype
    # 在2 ** l域的0分享
    x_l = ArithmeticSharedTensor.PRZS(size, device=device).share

    if isinstance(x_l, CUDALongTensor):
        x_l = x_l.tensor()

    rank = comm.get().get_rank()
    # 两个生成器，用于相减生成0的分享
    gen0 = comm.get().get_generator(0, device=device)
    gen1 = comm.get().get_generator(1, device=device)

    # 为了debug时在1、2进程中不产生错误
    x = 0
    x_i_p = 0
    if rank == 0:
        # 随机生成的公开值
        x = torch.empty(size=size, dtype=dtype, device=device).random_(-(ring_size // 2), to=(ring_size - 1) // 2)

        # 每bit在质数域67的分享,从低位到高位的torch.tensor's
        # torch_stack:根据tensors里面是否有is_cuda的决定调用CUDALongTensor.stack还是torch.stack
        x_i_p = torch_stack([((x >> i) & 1).byte() for i in range(l)])

    #每bit在质数域67的分享,从低位到高位的list
    current_rand_tensor = torch_stack([torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=256,generator=gen0) for i in range(64)])
    next_rand_tensor = torch_stack([torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=256,generator=gen1) for i in range(64)])

    ss_x_i_p = current_rand_tensor - next_rand_tensor

    #只在rank=0处做随机化
    if rank == 0:
        x_l = x_l + x
        ss_x_i_p = ss_x_i_p +x_i_p

        # 模拟wrap3(x1,x2,x3,l)在二进制域下的分享:rank==0时，获取1，2的xi来做
        x_l_rep1 = torch.zeros_like(x_l.data)
        x_l_rep2 = torch.zeros_like(x_l.data)
        recv_group10 = getattr(comm.get(), f"group{1}{0}")
        recv_group20 = getattr(comm.get(), f"group{2}{0}")
        req1 = comm.get().irecv(x_l_rep1.data, src=1, group=recv_group10 )
        req2 = comm.get().irecv(x_l_rep2.data, src=2, group=recv_group20 )
        req1.wait()
        req2.wait()

        #实现本地的wrap3
        x_2=wrap3_local(x_l,x_l_rep1,x_l_rep2)

        # 在2进制域的分享
        send_group02 = getattr(comm.get(), f"group{0}{2}")
        req4 = comm.get().isend((x_2).data, dst=2, group=send_group02)
        req4.wait()
        x_2 = x_2^1
        send_group01 = getattr(comm.get(), f"group{0}{1}")
        req3 = comm.get().isend((x_2).data, dst=1, group=send_group01)
        req3.wait()
    elif rank == 1:
        x_2 = torch.zeros_like(x_l.data).byte()
        send_group = getattr(comm.get(), f"group{1}{0}")
        req1 = comm.get().isend(x_l.data, dst=0, group=send_group)
        req1.wait()
        recv_group01 = getattr(comm.get(), f"group{0}{1}")
        req2 = comm.get().irecv(x_2.data, src=0, group=recv_group01)
        req2.wait()
    else:
        x_2 = torch.zeros_like(x_l.data).byte()
        send_group = getattr(comm.get(), f"group{2}{0}")
        req1 = comm.get().isend(x_l.data, dst=0, group=send_group)
        req1.wait()
        recv_group02 = getattr(comm.get(), f"group{0}{2}")
        req2 = comm.get().irecv(x_2.data, src=0, group=recv_group02)
        req2.wait()

    # debug_in_func(x_l,ss_x_i_p,x_2,x,x_i_p)

    return x_l,ss_x_i_p,x_2;

def debug_in_func(x_l,ss_x_i_p,x_2,x,x_i_p):
    #检验正确性
    x_gen = comm.get().all_reduce(x_l)
    x_i_gen = comm.get().all_reduce(ss_x_i_p)
    x_2_gen = comm.get().all_reduce(x_2)&1
    rank = comm.get().get_rank()
    if rank == 0:
        #print(f'2进制域绕环判断成功={x_2_gen.equal(wrap3_local(x_l,x_l_rep1,x_l_rep2))}')
        print(f'l域是否构建成功={x.equal(x_gen)}')
        print(f'p域是否构建成功={x_i_p.equal(x_i_gen)}')
        # 验证bit转换为l域是否正确
        x_rebuild = torch.zeros_like(x_l.data)
        for i in range(0, l):
            x_rebuild += x_i_p[i].long() << i

        if (x_rebuild).equal(x):
            print('rebuild right')
        else:
            print('rebuild wrong')

"""本地两数绕环判断：等价于无符号数的溢出，但tensor64位整形必有符号，因此要做额外处理
   输入数据就是torch.tensor 
   输出2进制的分享就是dtype=torch.uint8' torch.tensor:为1就是对应位置发生绕环
   余和的输出就是dtype=dtype' torch.tensor
"""
def wrap2_local(x,y):
    assert x.size()==y.size(),"wrap2 has different size"
    #符号位的提取,为1为负
    x_msb = (x>>(l-1)).byte()
    y_msb = (y>>(l-1)).byte()


    #为1为有效的元素
    neg2 = x_msb&y_msb
    pos2 = (x_msb|y_msb)^1
    pos1_neg1 = 1-neg2-pos2

    #是否绕环、余和的输出
    wrap = 0+neg2


    #两正两负/一正一负不饶的余和，一正一负饶的余和
    p2_or_n2_or_p1n1_notwrap_mod_l = (x&num_and)+(y&num_and)
    p1n1_wrap_mod_l = p2_or_n2_or_p1n1_notwrap_mod_l&num_and

    #一正一负绕环为1，不饶为0
    p1n1_wrap = (p2_or_n2_or_p1n1_notwrap_mod_l>>(l-1)).byte()&pos1_neg1
    wrap += p1n1_wrap

    #输出余和
    sum_mod_l = (p1n1_wrap*p1n1_wrap_mod_l)+((1-p1n1_wrap)*p2_or_n2_or_p1n1_notwrap_mod_l)+((pos1_neg1-p1n1_wrap).long()*(-1<<63))


    return wrap,sum_mod_l


"""本地三数绕环判断：具体定义见falcon论文
   输入数据就是torch.tensor or CUDALongTensor
   输出2进制的分享就是dtype=torch.uint8' torch.tensor:为1就是对应位置发生绕环
"""
def wrap3_local(x0,x1,x2):
    # 转换为torch.tensor
    if isinstance(x0, CUDALongTensor):
        x0 = x0.tensor()
    if isinstance(x1, CUDALongTensor):
        x1 = x1.tensor()
    if isinstance(x2, CUDALongTensor):
        x2 = x2.tensor()
    wrap01,sum_mod_l01 = wrap2_local(x0,x1)
    wrap01_2,sum_mod_l01_2 = wrap2_local(sum_mod_l01,x2)
    return wrap01^wrap01_2
    # return ((1-wrap01)*wrap01_2)+(wrap01*(1 - wrap01_2))

""" 
    size应当是没有升维64的
    根据输入的size，产生一个对应size的随机tensor的2进制的秘密分享,和其在p=256下的bit秘密分享,结果按序返回;
    1 没有严格的遵守安全规范：要用第四方服务器来产生这个随机值，而是和crypten一样，在第一方本地生成
    2 bit的分享就是dtype=torch.uint8' torch.tensor or CUDALongTensor
    3 2进制的分享就是dtype=torch.uint8' torch.tensor
"""
@timer_precomq
def generate_2_p_share(size,device='cuda'):
    # print('come generate_2_p_share')
    rank = comm.get().get_rank()
    # 两个生成器，用于相减生成0的分享
    gen0 = comm.get().get_generator(0, device=device)
    gen1 = comm.get().get_generator(1, device=device)
    group01 = getattr(comm.get(), f"group{0}{1}")
    group02 = getattr(comm.get(), f"group{0}{2}")

    # 每bit在质数域67的分享,从低位到高位的list
    current_rand_tensor = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=256, generator=gen0)
    next_rand_tensor = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=256, generator=gen1)
    beta_p = current_rand_tensor - next_rand_tensor
    beta = torch.zeros(size=size, dtype=torch.uint8, device=device)
    

    if rank == 0:
        beta = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=2)
        beta_p = beta_p + beta
        r1 = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=2)
        r2 = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=2)
        beta0 = beta^r1
        beta1 = beta^r2
        beta2 = beta^r2^r1
        #****
        oldbeta = beta
        #****
        beta = beta0
        #发给rank=1,rank=2
        req1 = comm.get().isend((beta1).data, dst=1, group=group01)
        req2 = comm.get().isend((beta2).data, dst=2, group=group02)
        req1.wait()
        req2.wait()
    if rank == 1:
        beta = torch.zeros(size=size, dtype=torch.uint8, device=device).data
        req1 = comm.get().irecv(beta, src=0, group=group01)
        req1.wait()
    if rank == 2:
        beta = torch.zeros(size=size, dtype=torch.uint8, device=device).data
        req1 = comm.get().irecv(beta, src=0, group=group02)
        req1.wait()
    # beta_gen = comm.get().all_reduce(beta)&1
    # beta_p_gen = comm.get().all_reduce(beta_p)
    # # print(f'beta_gen={beta_gen}')
    # # print(beta_p)
    # if rank == 0:
    #     if beta_gen.equal(beta_p_gen):
    #         print('beta equal')
    #     else:
    #         print('beta not equal')
    #     # print(beta_gen)
    #     # print(beta_p_gen)
    #     if beta_gen.equal(oldbeta) and beta_p_gen.equal(oldbeta):
    #         print('rebuild right')
    #     else:
    #         print('rebuild wrong')
    beta_p = torch_stack([beta_p for i in range(64)])
    #beta = torch_stack([beta for i in range(64)])
    #print(beta.size())
    return beta,beta_p


""" 
    根据输入的size，产生一个对应size的随机tensor的2进制的秘密分享,和其在l=2<<64的整数秘密分享,结果按序返回;
    1 没有严格的遵守安全规范：要用第四方服务器来产生这个随机值，而是和crypten一样，在第一方本地生成
    2 bit的分享就是dtype=torch.uint8' torch.tensor
    3 整数秘密分享就是dtype=torch.int56' torch.tensor
"""
@timer_precomq
def generate_2_l_share(size,device='cuda'):
    rank = comm.get().get_rank()
    c_l = ArithmeticSharedTensor.PRZS(size, device=device).share
    if isinstance(c_l,CUDALongTensor):
        c_l = c_l.tensor()
    group01 = getattr(comm.get(), f"group{0}{1}")
    group02 = getattr(comm.get(), f"group{0}{2}")

    #线程0模拟第四台服务器产生随机bit的布尔分享和64位整数秘密分享
    if rank == 0:
        c = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=2)
        r1 = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=2)
        r2 = torch.empty(size=size, dtype=torch.uint8, device=device).random_(0, to=2)
        c0 = c^r1
        c1 = c^r2
        c2 = c^r2^r1
        c_l = c_l + c
        c_2 = c0
        #发给rank=1,rank=2
        req1 = comm.get().isend((c1).data, dst=1, group=group01)
        req2 = comm.get().isend((c2).data, dst=2, group=group02)
        req1.wait()
        req2.wait()
    if rank == 1:
        c_2 = torch.zeros(size=size, dtype=torch.uint8, device=device).data
        req1 = comm.get().irecv(c_2, src=0, group=group01)
        req1.wait()
    if rank == 2:
        c_2 = torch.zeros(size=size, dtype=torch.uint8, device=device).data
        req1 = comm.get().irecv(c_2, src=0, group=group02)
        req1.wait()

    return c_2,c_l





""" 测试正确性的部分 """
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# size = [2,3]
# size_bit = size.copy()
# size_bit.insert(0,64)
# device = 'cpu'
#
# def debug_cpu_withoutdict():
#     global size,device
#     device = 'cpu'
#     launcher = multiprocess_launcher.MultiProcessLauncher(
#         3, func1_cpu_withoutdict
#     )
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
#
#
# def debug_cpu(dict):
#     global size,device
#     manager = Manager()
#     dict = manager.dict()
#     dict['x_gen'] = torch.zeros(size=size, device=device, dtype=dtype)
#     dict['x_bit_gen'] = torch.zeros(size=size_bit, device=device, dtype=torch.uint8)
#     device = 'cpu'
#     launcher = multiprocess_launcher.MultiProcessLauncher(
#         3, func1_cpu,fn_args=dict
#     )
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
#     print(f"x_gen={dict['x_gen']} x_true={dict['x_true']}")
#     #print(f"x_bit_gen={dict['x_bit_gen']} x_true={dict['x_bit_true']}")
#     if (dict['x_gen']).equal(dict['x_true']):
#         print('l right')
#     else:
#         print('l wrong')
#     if (dict['x_bit_gen']).equal(dict['x_bit_true']):
#         print('p right')
#     else:
#         print('p wrong')
#     print(wrap3_local(dict['x1'],dict['x2'],dict['x3']))
#
#
# def debug_gpu():
#     global size,device
#     # 此处修改无用
#     device = 'cuda'
#     launcher = multiprocess_launcher.MultiProcessLauncher(
#         3, func1_gpu,
#     )
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
#
#
# def func1_cpu_withoutdict():
#     generate_random_ring_element_bits_share_wrap3(size=size,device=device)
#
#
# def func1_cpu(dict):
#     comm.get().set_verbosity(True)
#     name = multiprocessing.process.current_process().name
#     xi, x, xi_bit, x_bit,x_2= generate_random_ring_element_bits_share_wrap3(size=size,device=device)
#     if(isinstance(x,CUDALongTensor)):
#         x = x.tensor
#     xi = xi
#     if isinstance(xi,CUDALongTensor):
#         xi = xi.tensor
#     rank = comm.get().get_rank()
#     if rank == 0:
#         dict['x1']=xi
#     elif rank == 1:
#         dict['x2']=xi
#     else:
#         dict['x3']=xi
#     print(f"{name}  :x_2={x_2}")
#     dict['x_gen'] += xi
#     dict['x_bit_gen'] = (dict['x_bit_gen']+xi_bit)%p
#     if name == 'process 0':
#         #print(f"x={x}")
#         dict['x_true'] = x
#         dict['x_bit_true'] = x_bit
#
#     if name == 'process 0':
#         # 验证bit转换为l域是否正确
#         x_rebuild = torch.zeros(size=size, dtype=dtype, device=device)
#         for i in range(0, l):
#             x_rebuild += x_bit[i].long() << i
#
#         #print(x_rebuild)
#         #print(f'x_true={x_true}')
#         if (x_rebuild).equal(dict['x_true']):
#             print('rebuild right')
#         else:
#             print('rebuild wrong')
#
#
# def func1_gpu():
#     global size,device
#     name = multiprocessing.process.current_process().name
#     xi, x, xi_bit, x_bit ,x_2= generate_random_ring_element_bits_share_wrap3(size=size,device=device)
#
#
# def debug_wrap2_local():
#     x = torch.tensor([4,-1<<63,3,2], dtype=dtype, device=device)
#     y = torch.tensor([10, -1,-1,-3], dtype=dtype, device=device)
#
#     o1,o2 = wrap2_local(x,y)
#     print(o1)
#     print(o2)
#     x = torch.randint(low=0,high=10, size=[10000,1000], dtype=dtype, device=device)
#     y = torch.randint(low=0, high=10, size=[10000, 1000], dtype=dtype, device=device)
#     o1, o2 = wrap2_local(x, y)
#
# def debug_wrap3_local():
#     x0 = torch.tensor([-1,13,7,-3], dtype=dtype, device=device)#[-67,-52,-73,-69]
#     x1 = torch.tensor([-1,-10,3,2], dtype=dtype, device=device)
#     x2 = torch.tensor([-2, 2,-1,-3], dtype=dtype, device=device)
#     wrap01,res= wrap2_local(x0,x1)
#     wrap_res,tmp = wrap2_local(res,x2)
#     print(f'wrpa01={wrap01}')
#     print(f'wrpa_res={wrap_res}')
#     print(f'wrpa3={wrap3_local(x0,x1,x2)}')
#
#
# def debug_generate_2_p_share():
#     launcher = multiprocess_launcher.MultiProcessLauncher(
#         3, func3
#     )
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
#
#
# def func3():
#     rank = ank = comm.get().get_rank()
#     beta,beta_p = generate_2_p_share(size=size,device=device)
#     beta_p = beta_p[0]
#     beta = comm.get().all_reduce(beta)&1
#     beta_p_gen = comm.get().all_reduce(beta_p)%p
#     if rank == 0:
#         if beta.equal(beta_p_gen):
#             print("right")
#         else:
#             print('wrong')
#
#
#
# if __name__ == '__main__':
#      debug_wrap3_local()