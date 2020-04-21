#ifndef POS_H
#define POS_H
# pragma once
# include <vector>
# include <cmath>
# define MAX_DIS 1000000000
using std::vector;
//表示一个城市的坐标信息
typedef struct {
	double x_index;
	double y_index;
}proxy;
class PSO {
private:
	//定义交换集
	typedef struct {
		size_t ei;
		size_t ej;
	}exchangeSeq;
	//记录粒子的路径信息
	typedef struct {
		vector<size_t> sbest;
		double path_length;
	}seqlen;
	//记录粒子的速度以及最好的位置信息
	typedef struct {
		vector<size_t> idl;// 粒子的当前位置信息
		vector<exchangeSeq> velocity;//粒子当前的速度信息
		seqlen pbest;//粒子最好的位置信息
	}particle;
	//定义基本的参数
	size_t popsize;//种群数量
	size_t genMax;//定义迭代的次数
	//定义每个迭代过程中的例子位置信息，
	//速度信息以及每个粒子在每一个迭代过程中的最好的位置信息
	vector<particle> particleSwarm;
	seqlen gbest;//全局最好的长度信息以及位置信息
public:
	//构造函数
	PSO(size_t popsize, size_t genMax) {
		this->genMax = genMax;
		this->popsize = popsize;
	}
	//初始化粒子群
	void InitialSwarm(vector<vector<double> >&citys_distance);
	//计算出距离矩阵
	static vector<vector<double> > CalculateDistance(vector<proxy>& citys_mat);
	// 计算一条路径的长度
	static double GetPathLength(vector<vector<double> >&citys_distance, const vector<size_t>& path_seq);
	//产生随机序列信息
	static vector<size_t> BuildRandomSequence(size_t length);
	//打印出粒子的路径和最短距离信息
	void print();

	

	//析构函数
	~PSO();
};
#endif // !POS_H