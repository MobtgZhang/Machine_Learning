import numpy as np
# 定义交换子
class exchangeSet:
    def __init__(self, ei, ej):
        self.ei = ei
        self.ej = ej
    def __str__(self):
        return str((self.ei, self.ej))
    def __repr__(self):
        return str((self.ei, self.ej))
# 定义交换序列
class exchangeSeq:
    def __init__(self,seq=None):
        if seq is not None and type(seq) == list:
            self.exchange_list = seq
        self.exchange_list = []
    def __str__(self):
        return str(self.exchange_list)
    def __repr__(self):
        return str(self.exchange_list)
    def append(self,exset):
        if type(exset) == exchangeSet:
            self.exchange_list.append(exset)
        elif type(exset) == list and len(exset)==2:
            self.exchange_list.append(exchangeSet(*exset))
        elif type(exset) == tuple and len(exset)==2:
            self.exchange_list.append(exchangeSet(*exset))
        else:
            raise TypeError("Error for model type or value :%s"%str(type(exset)))
    def __len__(self):
        return len(self.exchange_list)
    def __getitem__(self, item):
        return self.exchange_list[item]
    def __neg__(self):
        new_list = []
        length = len(self.exchange_list)
        for k in range(length-1,-1,-1):
            new_list.append(self.exchange_list[k])
        return exchangeSeq(new_list)
    def __add__(self, other):
        return exchangeSeq.computeEquivalentSet(self,other)
    def max(self):
        '''
        求出交换序中元素的最大值
        :return:
        '''
        length_num = 0
        for k in range(len(self.exchange_list)):
            length_num = max(self.exchange_list[k].ei,self.exchange_list[k].ej,length_num)
        return length_num+1
    def copy(self,seq = None):
        '''
        交换算子之间的复制操作
        :param seq:
        :return:
        '''
        if seq is None:
            new_seq = exchangeSeq()
            for k in range(len(self.exchange_list)):
                new_seq.append((self.exchange_list[k].ei,self.exchange_list[k].ej))
            return new_seq
        elif type(seq) == exchangeSeq:
            self.exchange_list = []
            for k in range(len(seq)):
                self.append((seq[k].ei,seq[k].ej))
        else:
            raise TypeError("Error for model type or value :%s" % str(type(seq)))
    @staticmethod
    def buildBasicExchangeSeq(seq1,seq2):
        '''
        根据两个序列，生成基本的交换序集
        :param seq1: type of Sequence
        :param seq2: type of Sequence
        :return Seq: type of exchangeSeq
        '''
        Seq = exchangeSeq()
        seqCopy = seq2.copy()
        seq_len = len(seq1)
        for k in range(seq_len):
            index = k
            while(index<seq_len and seqCopy[index]!=seq1[k]):
                index += 1
            if k == index:
                continue
            else:
                tmp = exchangeSet(k, index)
                seqCopy[index],seqCopy[k] = seqCopy[k],seqCopy[index]
                Seq.append(tmp)
        return Seq
    @staticmethod
    def computeEquivalentSet(sq1,sq2):
        '''
        根据两个交换集，生成对应的等价交换集
        seq1 ==> seq2  ---> out1
        seq2 = seq1+out1
        seq2 ==> seq3  ---> out2
        seq3 = seq2+out2
        seq1 ==> seq3  ---> out
        seq3 = seq1 + out
        所以有
        seq3 = seq2+out2 = seq1 + out1 + out2
        定义 out = out1 + out2,其中 seq3 = seq1 + out
        所以得到
        out = seq3 - seq1
        :param sq1:
        :param sq2:
        :return:
        '''
        if type(sq1) == type(sq2) and type(sq1) == exchangeSeq:
            seq_len = max(sq1.max(),sq2.max())
            # 创建随机排列
            seq1 = Sequence(seq_length=seq_len)
            seq2 = seq1 + sq1
            seq2 = seq2 + sq2
            return seq2-seq1
        else:
            raise TypeError("The type:%s and type:%s don't match!" %(str(type(sq1)),str(type(sq2))))
    @staticmethod
    def computeNextPos(x_position,seq):
        '''
        根据当前的序列和交换序，求出下一个序列
        :param x_position: type of Sequence
        :param seq: type of exchangeSeq or exchangeSet
        :return:
        '''
        if type(x_position) == Sequence:
            y_position = x_position.copy()
            if type(seq) == exchangeSeq:
                citys_sum = len(seq)
                for k in range(citys_sum):
                    y_position[seq[k].ei],y_position[seq[k].ej] = y_position[seq[k].ej],y_position[seq[k].ei]
                return y_position
            elif type(seq) == exchangeSet:
                y_position[seq.ei], y_position[seq.ej] = y_position[seq.ej], y_position[seq.ei]
                return y_position
            else:
                raise TypeError("Unknown seq type: %s" % str(type(seq)))
        else:
            raise TypeError("Unknown x_position type: %s" % str(type(x_position)))

    @staticmethod
    def computePreviousPos(x_position, seq):
        '''
        根据当前的序列和交换序，求出上一个序列
        :param x_position: type of Sequence
        :param seq: type of exchangeSeq or exchangeSet
        :return:
        '''
        if type(x_position) == Sequence:
            y_position = x_position.copy()
            if type(seq) == exchangeSeq:
                citys_sum = len(seq)
                for k in range(citys_sum-1,-1,-1):
                    y_position[seq[k].ei], y_position[seq[k].ej] = y_position[seq[k].ej], y_position[seq[k].ei]
                return y_position
            elif type(seq) == exchangeSet:
                y_position[seq.ei], y_position[seq.ej] = y_position[seq.ej], y_position[seq.ei]
                return y_position
            else:
                raise TypeError("Unknown seq type: %s" % str(type(seq)))
        else:
            raise TypeError("Unknown x_position type: %s" % str(type(x_position)))

class Sequence:
    def __init__(self,seq = None,seq_length = None):
        self.seq_length = seq_length
        self.sequence = seq
        if seq is not None and type(seq) == Sequence:
            self.copy(seq)
        elif type(seq) == np.ndarray and seq.ndim == 1:
            self.sequence = seq.copy()
            self.seq_length = len(seq)
        elif seq_length is not None and type(seq_length) == int:
            self.sequence = np.random.permutation(seq_length)
            self.seq_length = seq_length
        else:
            pass
    def __add__(self, other):
        '''
        定义序列与一个交换序列相加的操作，得到另一个序列(也就是下一个序列)
        :param other: type of  exchangeSeq
        :return:
        '''
        if type(other) == exchangeSeq:
            out = exchangeSeq.computeNextPos(self,other)
            return out
        else:
            raise TypeError("Unknown type: %s" % str(type(other)))
    def __sub__(self, other):
        '''
        定义两个交换序列相减的操作，得到一个交换序列
        :param other: type of  exchangeSeq
        :return:
        '''
        if type(other) == Sequence:
            out = exchangeSeq.buildBasicExchangeSeq(self,other)
            return out
        elif type(other) == exchangeSeq:
            out = exchangeSeq.computePreviousPos(self, other)
            return out
        else:
            raise TypeError("Unknown type: %s" % str(type(other)))
    def __len__(self):
        return self.seq_length
    def __getitem__(self, item):
        return self.sequence[item]
    def __setitem__(self, key, value):
        self.sequence[key] = value
    def __str__(self):
        return str(self.sequence)
    def copy(self,seq = None):
        if type(seq) == Sequence:
            self.sequence = seq.sequence.copy()
            self.seq_length = len(seq)
        elif seq is None:
            return Sequence(seq=self)
        else:
            raise TypeError("Unknown type: %s"%str(type(seq)))
