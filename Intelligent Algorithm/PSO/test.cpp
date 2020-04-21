
#pragma once
#include "PSO.h"
#include"util.h"
int main()
{
	string filename = "eil51.tsp";
	string root_path = "Data";
	download(root_path, filename);
	vector<proxy>OutData = getData(root_path + "\\" + filename);
	size_t popsize = 600;
	size_t genMax = 80;
	PSO model = PSO(popsize, genMax);
	//初始化粒子群
	vector<vector<double> >citys_distance = PSO::CalculateDistance(OutData);
	model.InitialSwarm(citys_distance);
	//粒子进行移动
	model.Move(citys_distance);
	//打印函数
	model.print();
	system("pause");
	return 0;
}