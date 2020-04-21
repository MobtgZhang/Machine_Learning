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

	//生成基本交换序
	vector<exchangeSeq> BuildBasicExchangeSeq(vector<size_t>seq1, vector<size_t>seq2);
	//根据当前解计算下一个解，也即下一个位置
	void computeNextPos(vector<size_t>&id1, vector<exchangeSeq>&v);
	//生成等价集
	vector<exchangeSeq> computeEquivalentSet(vector<exchangeSeq> sq1, vector<exchangeSeq> sq2);



	//每一个迭代过程中的移动方法
	void Movement(vector<vector<double> >citys_distance);
	//粒子进行移动
	void PSO::Move(vector<vector<double> >citys_distance);

	//根据当前粒子的信息，计算更新速度
	void computeNewVelocity(particle& pl);

	//析构函数
	~PSO();
};
#endif // !POS_H