#pragma once
#ifndef DNN_H
#define DNN_H
#include<vector>
#include "matrix.h"
class DNN {
private:
	std::vector<unsigned int>layers_list;
public:
	//构造函数
	DNN();
	//前向传播过程

	//析构函数
	~DNN();
};
#endif // !DNN_H
#ifndef BP_NET_H
#define BP_NET_H
class BpNet {
private:
	unsigned int in_dim;
	unsigned int hid_dim;
	unsigned int out_dim;
	Matrix<double>WeightIn;
	Matrix<double>BaisIn;
	Matrix<double>WeightOut;
	Matrix<double>BaisOut;
public:
	//构造函数
	BpNet(unsigned int in_dim, unsigned int hid_dim, unsigned int out_dim);
	//前向传播过程
	Matrix<double> forward(Matrix<double>&input);
	//反向传播过程
	//更新梯度
	//析构函数
	~BpNet();
};
//构造函数的实现过程
BpNet::BpNet(unsigned int in_dim, unsigned int hid_dim, unsigned int out_dim) {
	this->in_dim = in_dim;
	this->hid_dim = hid_dim;
	this->out_dim = out_dim;
	//定义矩阵和偏置
	
}
#endif // !BP_NET_H

