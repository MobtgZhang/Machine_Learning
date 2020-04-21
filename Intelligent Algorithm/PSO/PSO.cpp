#pragma once
#include "PSO.h"
#include<iostream>
using std::cout;
using std::endl;
#include<string>
#include<sstream>
using std::string;
using std::stringstream;
string toString(vector<size_t> arr) {
	string line = "";
	stringstream ss;
	string tmp;
	for (size_t k = 0; k < arr.size(); k++) {
		ss << arr[k];
		tmp = ss.str();
		ss.str("");
		line = line + tmp + " ";
	}
	return line;
}
//析构函数
PSO::~PSO() {

}
# include<ctime>
#include<cstdlib>
// 计算一条路径的长度
double PSO::GetPathLength(vector<vector<double> >&citys_distance, const vector<size_t>&path_seq) {
	double path_length = 0;
	size_t citys_num = citys_distance.size();
	for (size_t pos = 1; pos < citys_num; pos++) {
		path_length += citys_distance[path_seq[pos-1]][pos];
	}
	path_length += citys_distance[path_seq[citys_num-1]][0];
	return path_length;
}
//计算出距离矩阵
vector<vector<double> > PSO::CalculateDistance(vector<proxy>& citys_mat) {
	vector<vector<double> > citys_distance;
	//获取到城市的个数信息
	unsigned int citys_num = citys_mat.size();
	//初始化距离矩阵
	for (size_t k = 0; k < citys_num; k++) {
		citys_distance.push_back(vector<double>());
		for (size_t j = 0; j < citys_num; j++) {
			citys_distance[k].push_back(sqrt(pow(citys_mat[k].x_index - citys_mat[j].x_index, 2) + pow(citys_mat[k].y_index - citys_mat[j].y_index, 2)));
		}
	}
	return citys_distance;
}

//产生随机序列信息
vector<size_t> PSO:: BuildRandomSequence(size_t length) {
	vector<size_t> vc;
	for (size_t i = 0; i < length; i++) vc.push_back(i);
	for (size_t i = length - 1; i > 0; i--) {
		size_t x = rand() % (i + 1);
		size_t tmp = vc[i];
		vc[i] = vc[x];
		vc[x] = tmp;
	}
	return vc;
}


//初始化粒子群
void PSO::InitialSwarm(vector<vector<double> >&citys_distance) {
	size_t tp = 0;
	particle pt;////产生粒子当前的速度以及位置和最好的位置信息
	exchangeSeq exSeq;//位置的交换子
	double shortlen = MAX_DIS;//最小的长度
	size_t citys_num= citys_distance.size();
	srand((unsigned)time(NULL));
	for (size_t i = 0; i < this->popsize; i++) {
		pt.idl = PSO::BuildRandomSequence(citys_num);//初始化最初的位置信息
		pt.pbest.sbest = pt.idl;
		pt.pbest.path_length = PSO::GetPathLength(citys_distance,pt.pbest.sbest);
		for (size_t j = 0; j < citys_num; j++) {
			//产生随机交换子
			exSeq.ei = rand() % citys_num;
			exSeq.ej = rand() % citys_num;
			pt.velocity.push_back(exSeq);
		}
		this->particleSwarm.push_back(pt);
		if (shortlen > pt.pbest.path_length) {
			//产生最小的长度
			shortlen = pt.pbest.path_length;
			tp = i;
		}
	}
	this->gbest = this->particleSwarm[tp].pbest;
}
//打印出结果，即最短路径和距离信息
void PSO::print() {
	cout << "最短的距离为: " << this->gbest.path_length << endl;
	cout << "最短的路径信息为: ";
	for (size_t i = 0; i < gbest.sbest.size(); i++) {
		cout << gbest.sbest[i] << "\t";
	}
	cout << gbest.sbest[0] << endl;
}
//生成基本交换序
vector<PSO::exchangeSeq> PSO::BuildBasicExchangeSeq(vector<size_t>seq1, vector<size_t>seq2) {
	vector<exchangeSeq>Seq;
	vector<size_t> sq = seq2;
	PSO::exchangeSeq q;
	size_t i, j;
	for (i = 0; i < seq1.size()-1; i++) {
		for (j = i; j < seq1.size() && sq[j] != seq1[i]; j++);
		q.ei = i;
		q.ej = j;
		if (i == j) continue;
		size_t tp = sq[i];
		sq[i] = sq[j];
		sq[j] = tp;
		Seq.push_back(q);
	}
	return Seq;
}
//生成等价集
vector<PSO::exchangeSeq> PSO::computeEquivalentSet(vector<PSO::exchangeSeq> sq1, vector<PSO::exchangeSeq> sq2) {
	vector<size_t> seq1, seq2;
	size_t citys_num = sq1.size();
	for (size_t i = 0; i < citys_num; i++) seq1.push_back(i);
	seq2 = seq1;
	computeNextPos(seq1, sq1);
	computeNextPos(seq1, sq2);
	return this->BuildBasicExchangeSeq(seq1,seq2);
}
//根据当前解计算下一个解，也即下一个位置
void PSO::computeNextPos(vector<size_t>&id1, vector<PSO::exchangeSeq>&v){
	for (size_t i = 0; i < v.size(); i++) {
		size_t tp = id1[v[i].ei];
		id1[v[i].ei] = id1[v[i].ej];
		id1[v[i].ej] = tp;
	}
}
//根据当前粒子的信息，计算更新速度
void PSO::computeNewVelocity(PSO::particle& pl) {
	//对每一个个体更新速度信息
	vector<PSO::exchangeSeq> Pid = BuildBasicExchangeSeq(pl.pbest.sbest,pl.idl);
	//全局更新个体信息
	vector<PSO::exchangeSeq> Pgd = BuildBasicExchangeSeq(gbest.sbest, pl.idl);
	vector<PSO::exchangeSeq> tp = pl.velocity;
	//生成对应的等价集
	tp = this->computeEquivalentSet(tp,Pid);
	pl.velocity = this->computeEquivalentSet(tp, Pgd);
}


//每一个迭代过程中的移动方法
void PSO::Movement(vector<vector<double> >citys_distance) {
	size_t tp = 0;
	for (size_t i = 0; i < this->particleSwarm.size(); i++) {
		cout << toString(particleSwarm[i].idl) << endl;
		this->computeNextPos(this->particleSwarm[i].idl, particleSwarm[i].velocity);
		cout << toString(particleSwarm[i].idl) << endl;
		
		system("pause");
		this->computeNewVelocity(this->particleSwarm[i]);
		
		if (this->particleSwarm[i].pbest.path_length > PSO::GetPathLength(citys_distance, this->particleSwarm[i].idl)) {
			this->particleSwarm[i].pbest.path_length = PSO::GetPathLength(citys_distance, this->particleSwarm[i].idl);
			
			
			this->particleSwarm[i].pbest.sbest = this->particleSwarm[i].idl;
		}
		
		if (this->particleSwarm[i].pbest.path_length < this->gbest.path_length) {
			gbest.path_length = this->particleSwarm[i].pbest.path_length;
			tp = i;
		}
	}
	gbest.sbest = this->particleSwarm[tp].idl;
}
//粒子进行移动
void PSO::Move(vector<vector<double> >citys_distance) {
	for (size_t t= 0; t < this->genMax; t++) {
		this->Movement(citys_distance);
		cout << "Best length: " << this->gbest.path_length<<endl;
	}
}