#pragma once
#ifndef UTIL_H
#define UTIL_H
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <urlmon.h>
#include <fileapi.h>
#include <vector>
#pragma comment(lib, "urlmon.lib")
#include"PSO.h"

using std::string;
using std::cout;
using std::endl;
using std::ifstream;
using std::getline;
using std::vector;
using std::stringstream;


//定义下载文件函数
void download(string root_path, string filename) {
	//判断根目录是否存在，若不存在则创建根目录
	string url = "http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/";
	string data_path = root_path + "\\" + filename;
	DWORD dwAttribute = GetFileAttributes(root_path.c_str());
	if (dwAttribute == 0XFFFFFFFF) {
		//0XFFFFFFFF表示文件夹不存在,创建文件夹
		CreateDirectory(root_path.c_str(), NULL);
		//开始下载文件，并保存文件
		url = url + filename;
		DWORD status = URLDownloadToFile(NULL, url.c_str(), data_path.c_str(), 0, 0);
		if (status == S_OK) {
			cout << "Successfully download file: " << data_path << endl;
		}
		else {
			string message = "Download file: " + data_path + " failed!";
			throw message.c_str();
		}
	}
	else {
		//查看文件是否存在
		dwAttribute = GetFileAttributes(data_path.c_str());
		if (dwAttribute == 0XFFFFFFFF) {
			//开始下载文件，并保存文件
			url = url + filename;
			DWORD status = URLDownloadToFile(NULL, url.c_str(), data_path.c_str(), 0, 0);
			if (status == S_OK) {
				cout << "Successfully download file: " << data_path << endl;
			}
			else {
				string message = "Download file: " + data_path + " failed!";
				throw message.c_str();
			}

		}
		else {
			cout << "The file: " << data_path << " exisits!" << endl;
		}
	}
}
// 定义数据获取函数，用于提取文件中的城市数据信息
vector<proxy> getData(string filename) {
	ifstream infile(filename);
	
	vector<proxy> citys_math;
	//打开文件，并读取每一行的数据流
	string  line; //保存读入的每一行
	bool NumFlag = false;
	while (getline(infile, line)) {
		if (line.find("NODE_COORD_SECTION") != line.npos) {
			NumFlag = true;
		}
		else if(line.find("EOF") != line.npos) {
			break;
		}
		else if (NumFlag) {
			double index;
			double x_index;
			double y_index;
			stringstream stream;
			stream << line;
			stream >> index >> x_index >> y_index;
			proxy tmp;
			tmp.x_index = x_index;
			tmp.y_index = y_index;
			citys_math.push_back(tmp);
		}
		else {
			
		}
	}
	return citys_math;
}
#endif