from crypten.common.util import torch_stack,torch_cat

from crypten.mpc.primitives import ArithmeticSharedTensor

from crypten.cuda import CUDALongTensor

from crypten.mpc.primitives.resharing import replicate_shares
# from scripts import multiprocess_launcher

import torch
from crypten import communicator as comm
from crypten.mpc.gw_relu_helper import gw_get_msb
from crypten.encoder import FixedPointEncoder


"""先做分解，分解后size与最终maxpool后size一致，即维数一致
    orignal_tensor为torch.tensor int64
    返回的list中元素前者为ArithmeticSharedTensor,后者为 torch.tensor int64
"""
def decompose_2D(orignal_tensor,kernel_size,stride):
    # 获取最低二维的行列数
    # print(orignal_tensor.size())
    if  orignal_tensor.dim()==4:
        num1,num2,rows,clows = orignal_tensor.size()
        num_pictrue = num1*num2
    else:
        num_pictrue,rows,clows = orignal_tensor.size()
    # 获取步长
    stride_x,stride_y = stride

    # 获取maxpool后的size最低二维
    x_len = (rows - kernel_size[0])//stride_x + 1
    y_len = (clows - kernel_size[1])//stride_y + 1

    #获取orignal_tensor flatten后的index
    indexs = list()
    tensors_after_max_pool = list()
    # 获取orignal_tensor flatten后的每次max核的左上角起点位置,新的图片要重新计数
    index0 = torch.LongTensor([num*rows*clows + x*stride_x*clows + y*stride_y for num in range(num_pictrue) for x in range(x_len) for y in range(y_len)],device='cpu')
    index0 = index0.to(device=orignal_tensor.device)
    # 获取max核剩余坐标
    for x in range(kernel_size[0]):
        for y in range(kernel_size[1]):
            index = index0 + x * clows + y
            indexs.append(index)
            tensors_after_max_pool.append(orignal_tensor.take(index))

    #新的图片要重新计数
    # for i in range(len(indexs)):
        # cur_index = indexs[i].view([num_pictrue, x_len, y_len])
        # for j in range(1,num_pictrue):
            # cur_index[j] = cur_index[j] - j*rows*clows
        # indexs[i] = cur_index
    return tensors_after_max_pool,indexs


""" 
基于不经意传输的原理，符号位为1选cur，为0选max
max, cur, max_index,cur_index 为 ArithmeticSharedtensor
sign_bit 为torch.tensor uint8 
返回ArithmeticSharedtensor
"""
def ss_ot(sign_bit, max, cur,max_index,cur_index):
    sign_bit = sign_bit.long()
    sender, receiver, helper = [0, 1, 2]
    groups2r = getattr(comm.get(), f"group{sender}{receiver}")
    groups2h = getattr(comm.get(), f"group{sender}{helper}")
    groupr2h = getattr(comm.get(), f"group{receiver}{helper}")


    rank = comm.get().get_rank()
    zero_share = ArithmeticSharedTensor.PRZS(sign_bit.size(), device=sign_bit.device).share

    if rank == sender:
        xs = torch.ones_like(sign_bit.data)
        b1, b3 = sign_bit, replicate_shares(sign_bit)
        if isinstance(b3, CUDALongTensor):
            b3 = b3.tensor()

        b1, b3 = b1 & 1, b3 & 1

        # r 是为了让receiver无法重构信息
        ring_size = 2 ** 64
        r = torch.empty(size=sign_bit.size(), dtype=torch.long, device=sign_bit.device).random_(-(ring_size // 2),
                                                                                      to=(ring_size - 1) // 2)
        m0 = (b1 ^ b3 ^ 0) * xs - r
        m1 = (b1 ^ b3 ^ 1) * xs - r

        w0 = torch.empty(size=sign_bit.size(), dtype=torch.long, device=sign_bit.device).random_(-(ring_size // 2),
                                                                                       to=(ring_size - 1) // 2)
        w1 = torch.empty(size=sign_bit.size(), dtype=torch.long, device=sign_bit.device).random_(-(ring_size // 2),
                                                                                       to=(ring_size - 1) // 2)

        req0 = comm.get().isend(torch_stack([m0 ^ w0, m1 ^ w1]), dst=receiver, group=groups2r)
        req1 = comm.get().isend(torch_stack([w0, w1]), dst=helper, group=groups2h)
        req0.wait()
        req1.wait()

        a = ArithmeticSharedTensor.from_shares(zero_share + r, src=rank)

    if rank == receiver:
        b2, b1 = sign_bit, replicate_shares(sign_bit)
        if isinstance(b1, CUDALongTensor):
            b1 = b1.tensor()

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
        b3, b2 = sign_bit, replicate_shares(sign_bit)
        if isinstance(b2, CUDALongTensor):
            b2 = b2.tensor()

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

    a.encoder = FixedPointEncoder(precision_bits=None)
    scale = a.encoder._scale
    a *= scale

    diff_num = cur-max
    diff_index = cur_index - max_index
    out_num = (diff_num * a) + max
    out_index = (diff_index * a) + max_index
    #保证格式为torch.tensor int64
    # out_num = out_num.share
    # out_index = out_index.share
    # if isinstance(out_num,CUDALongTensor):
    #     out_num = out_num.tensor()
    # if isinstance(out_index , CUDALongTensor):
    #     out_index  = out_index .tensor()
    return out_num,out_index


"""2维的最大池化，kernel_size为2数的内核size，stride为2数的步长
    self为ArithmeticSharedTensor
    参与计算的也应该是是ArithmeticSharedTensor
    返回的max 是 ArithmeticSharedtensor
    max_index 强保密等级下是ArithmeticSharedtensor，弱保密等级下是torch.tensor int64
"""
def max_pool2d_falcon(self, kernel_size, padding=0, stride=None, return_indices=True):
    from crypten.mpc.primitives.converters import get_msb
    from crypten.mpc.gw_relu_helper import gw_get_msb
    import crypten
    # print('run max_pool2d_falcon')
    #保证参与计算的是普通tensor
    # tensor_torch = self.share
    # if isinstance(tensor_torch,CUDALongTensor):
    #     tensor_torch = tensor_torch.tensor()
    tensor_torch = self
    # print(tensor_torch)
    assert tensor_torch.dim()==4,'max_pool2d only support dim=4,but it can Scalable to support dim >4'

    #临时降低维度
    tensor_torch = tensor_torch.squeeze()

    #获取用于选择比较的list，和用于分享传播的indices
    tensors, indexs = decompose_2D(tensor_torch,kernel_size,stride)

    #num为内核长
    num = kernel_size[0]*kernel_size[1]

    #初始化
    tensors=stack(tensors)
    indexs=torch.stack(indexs)

    #若保密信息则将indexs转换为AithmeticSharedTensor
    #*************************************#
    indexs = ArithmeticSharedTensor(indexs)
    #*************************************#

    #进行num-1次类relu
    while tensors.size(0) > 1:
        
        m = tensors.size(0)
        x, y, remainder = tensors.split([m // 2, m // 2, m % 2], dim=0)
        x_i, y_i, remainder_i = indexs.split([m // 2, m // 2, m % 2], dim=0)
        diff = x - y

        # sign_bit = get_msb(diff)
        sign_bit = gw_get_msb(diff)

        #以下为不对内核元素大小情况进行保密的流程
        # max, max_index = compare_unprotect(sign_bit,max,tensors[i],max_index,indexs[i])

        # 以下为对内核元素大小情况进行保密的流程
        max, max_index = compare_protect(sign_bit,x,y,x_i, y_i)

        tensors = cat([max, remainder], dim=0)
        indexs = cat([max_index, remainder_i], dim=0)

    #修改tensor形状，以满足二维maxpool的要求
    # 获取最低二维的行列数
    if  tensor_torch.dim()==4:
        num1,num2,rows,clows = tensor_torch.size()
        num_pictrue = num1*num2
    else:
        num_pictrue,rows,clows = tensor_torch.size()

    # 获取步长
    stride_x,stride_y = stride

    # 获取maxpool后的size最低二维
    x_len = (rows - kernel_size[0])//stride_x + 1
    y_len = (clows - kernel_size[1])//stride_y + 1
    if  tensor_torch.dim()==4:
        max = max.view([num1,num2,x_len,y_len])

        max_index = max_index.view([num1,num2, x_len, y_len])
        # 新的图片要重新计数
        for i in range(0, num1):
            for j in range(0, num2):
                max_index[i][j] = max_index[i][j] - i*num2* rows * clows - j*rows * clows
    else:
        max = max.view([num_pictrue,x_len,y_len])

        max_index = max_index.view([num_pictrue, x_len, y_len])
        # 新的图片要重新计数
        for j in range(1, num_pictrue):
            max_index[j] = max_index[j] - j * rows * clows
        #提升维度，与总体保持一致
        max = max.unsqueeze(0)
        max_index = max_index.unsqueeze(0)
    
    return max,max_index


"""AirthmeticSharedtensor stack"""
def stack(tensors, *args, **kwargs):
    result = tensors[0].shallow_copy()
    result.share = torch_stack(
        [tensor.share for tensor in tensors], *args, **kwargs
    )
    return result


"""AirthmeticSharedtensor stack"""
def cat(tensors, *args, **kwargs):
    result = tensors[0].shallow_copy()
    result.share = torch_cat(
        [tensor.share for tensor in tensors], *args, **kwargs
    )
    return result


"""不对内核元素彼此大小情况进行保密的流程"""
def compare_unprotect(sign_bit,max,cur,index_max,indexs_cur):
    from crypten.mpc.primitives.binary import BinarySharedTensor
    #此安全等级，两元素大小关系可公开
    #为与aby3的获取符号位比较，做的兼容
    if isinstance(sign_bit,BinarySharedTensor):
        sign_bit = sign_bit.get_plain_text()
        sign_bit = sign_bit.byte()
    else:
        sign_bit = comm.get().all_reduce(sign_bit) & 1
    diff_num = cur-max
    diff_index = indexs_cur - index_max
    out_num = (diff_num * sign_bit) + max
    out_index = (diff_index * sign_bit) + index_max
    return out_num, out_index

"""对内核元素彼此大小情况进行保密的流程"""
def compare_protect(sign_bit,max,cur,index_max,index_cur):
    if isinstance(index_max,torch.Tensor):
        # tmp = ArithmeticSharedTensor.PRZS(index_max.size, device=index_max.device)
        # if comm.get().get_rank() == 0:
        #     tmp += index_max
        index_max = ArithmeticSharedTensor(index_max)
    if isinstance(index_cur, torch.Tensor):
        # tmp = ArithmeticSharedTensor.PRZS(index_cur.size, device=index_cur.device)
        # if comm.get().get_rank() == 0:
        #     tmp += index_cur
        index_cur = ArithmeticSharedTensor(index_cur)
    # print(f'index_max={index_max.get_plain_text(0)}')
    max,max_index = ss_ot(sign_bit,max,cur,index_max,index_cur)
    return max,max_index


"""maxpool的反向传播,转换为两个doubleTensor来做"""
def backward(grad_output, indices, kernel_size, stride):
    # 确保indices是普通的torch.tensor int64
    if not isinstance(indices, torch.Tensor):
        indices = indices.get_plain_text().to(dtype=torch.long, device=grad_output.device)

    # 保证是torch.tensor int64
    out = grad_output.share
    if not isinstance(indices, torch.Tensor):
        out = out.tensor()

    # 只在indices对应的位置填充，其余为0
    assert indices.dim() == 4, 'max_pool2d backward only support dim=4,but it can Scalable to support dim >4'

    #转换为两个doubleTensor来做
    block1,block2 = [(out >> (32 * i)) & (2 ** 32 - 1) for i in range(2)]
    block1, block2 =block1.double(),block2.double()

    max_unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride)

    # 满足调用UNmaxpoold的要求
    out1 = max_unpool(block1, indices)
    out1 = out1.long()
    out2 = max_unpool(block2, indices)
    out2 = out2.long()

    out = out1+(out2<<32)

    if out.device.type == 'cuda':
        out = CUDALongTensor(out)
    from crypten.mpc import MPCTensor
    out = MPCTensor.from_shares(out, src=comm.get().get_rank())

    return out
"""*****************************************************************************************************************************************"""
""" 测试正确性的部分 """
# def debug_decompose2D():
#     import crypten
#     a = torch.FloatTensor([[[10, 1, 8, 6], [2, 2, 2, 2], [61, 3, 3, 3], [8.5, 4.2, 4, 4]],
#                            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]])
#     # a = a.to(device='cuda')
#     a = crypten.cryptensor(a.to(dtype=torch.long), device="cpu", requires_grad=False)
#     a = a._tensor
#     tensors_after_max_pool, indexs = decompose_2D(a,kernel_size=(2, 2),stride=(2, 2))
#     for i in range(len(tensors_after_max_pool)):
#         tensors_after_max_pool[i] = tensors_after_max_pool[i].get_plain_text(0)
#     if comm.get().get_rank() == 0:
#         for i in range(len(tensors_after_max_pool)):
#             print(f'tensors[{i}]={tensors_after_max_pool[i]}')
#             print(f'index[{i}]={indexs[i]}')
#
#
# def debug_ss_ot():
#     pass
#
#
# def debug_max_pool2d_falcon():
#     import crypten
#     # import os
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     rank = comm.get().get_rank()
#     tensor_plain = torch.FloatTensor([[[[10,66,16,6],[2,2,2,2],[2,3,-1,3],[-1,4,4,9]],[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]]])
#     x = crypten.cryptensor(tensor_plain.to(dtype=torch.long), device="cuda", requires_grad=False)
#     x = x._tensor
#     kernel_size = (2, 2)
#     stride = (2, 2)
#     model = torch.nn.MaxPool2d(kernel_size=kernel_size,stride=stride,return_indices=True, padding=0)
#     out, indices = model(tensor_plain)
#     mpc_out, mpc_indices = max_pool2d_falcon(x,kernel_size=kernel_size,stride=stride)
#
#     unmaxpool,unmaxpool_mpc = debug_backward(out, indices,mpc_out, mpc_indices,kernel_size,stride)
#
#
#     mpc_out = mpc_out.get_plain_text(0)
#     if not isinstance( mpc_indices,torch.Tensor):
#         mpc_indices = mpc_indices.get_plain_text(0)
#
#     unmaxpool_mpc = unmaxpool_mpc.get_plain_text(0)
#     if rank == 0:
#         print(f'out={out}')
#         print(f'mpc_out={mpc_out}')
#         print(f'indices={indices}')
#         print(f'mpc_indices={mpc_indices}')
#         print(f'unmaxpool={unmaxpool}')
#         print(f'unmaxpool_mpc={unmaxpool_mpc}')
#
#
# def debug_backward(out, indices,mpc_out, mpc_indices,kernel_size,stride):
#     max_unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride)
#     out_unpool = max_unpool(out, indices)  # , output_size=a.size()
#     out_unpool = out_unpool.squeeze()
#     out_unpool_mpc = backward(mpc_out, mpc_indices, kernel_size, stride)
#     return out_unpool,out_unpool_mpc
#
# def debug_in_mpc():
#     from scripts import multiprocess_launcher
#     launcher = multiprocess_launcher.MultiProcessLauncher(
#         3, debug_max_pool2d_falcon
#     )
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
#
#
# if __name__ == '__main__':
#     debug_in_mpc()