#pragma once
#ifndef VECTOR_H
#define VECTOR_H
#include<string>
#include<sstream>
template<typename DataType>
class Vector {
private:
	unsigned int Length;
	DataType* InnerVec;
public:
	//构造函数
	Vector(unsigned int Length);
	Vector(Vector<DataType>&mat);
	//获取向量的大小
	unsigned int GetLength() { return this->Length;}
	//表示矩阵逐个元素相乘
	Vector<DataType> mul(Vector<DataType>& mat);
	Vector<DataType> mul(DataType value);
	//获取矩阵中的某一个元素
	DataType Get(unsigned int index);
	//设置矩阵中的某一个元素
	void Set(DataType value, unsigned int index);
	//表示矩阵逐个元素相加
	Vector<DataType> add(Vector<DataType>& mat);
	Vector<DataType> add(DataType value);
	//表示矩阵中逐个元素相减
	Vector<DataType> sub(Vector<DataType>& mat);
	Vector<DataType> sub(DataType value);
	//表示矩阵中逐个元素相除
	Vector<DataType> div(Vector<DataType>& mat);
	Vector<DataType> div(DataType value);
	//Vector 对Matrix广播
	Matrix<DataType> add(Matrix<DataType>& mat);
	Matrix<DataType> sub(Matrix<DataType>& mat);
	//重载运算符
	Vector<DataType> operator +(Vector<DataType>& mat) { return this->add(mat); }
	Vector<DataType> operator +(DataType value) { return this->add(value); }
	Vector<DataType> operator -(Vector<DataType>& mat) { return this->sub(mat); }
	Vector<DataType> operator -(DataType value) { return this->sub(value); }
	Vector<DataType> operator *(Vector<DataType>& mat) { return this->mul(mat); }
	Vector<DataType> operator *(DataType value) { return this->mul(value); }
	Vector<DataType> operator /(Vector<DataType>& mat) { return this->div(mat); }
	Vector<DataType> operator /(DataType value) { return this->div(value); }
	//重载运算符中的广播机制
	Matrix<DataType> operator +(Matrix<DataType>& mat) { return this->add(mat); }
	Matrix<DataType> operator -(Matrix<DataType>& mat) { return this->sub(mat); }
	//将矩阵中的元素转化为字符串形式
	std::string toString();
	//析构函数
	~Vector();
};
//构造函数
template<typename DataType>
Vector<DataType>::Vector(unsigned int Length) {
	this->Length = Length;
	this->InnerVec = new DataType[Length];
}
template<typename DataType>
Vector<DataType>::Vector(Vector<DataType>&mat) {
	this->Length = mat.GetLength();
	this->InnerVec = new DataType[Length];
	for (unsigned int k = 0; k < mat.GetLength(); k++) {
		this->InnerVec[k] = mat.Get(k);
	}
}
//析构函数
template<typename DataType>
Vector<DataType>::~Vector() {
	delete[] this->InnerVec;
}
//获取矩阵中的某一个元素
template<typename DataType>
DataType Vector<DataType>::Get(unsigned int index) {
	return this->InnerVec[index];
}
//设置矩阵中的某一个元素
template<typename DataType>
void Vector<DataType>::Set(DataType value, unsigned int index) {
	this->InnerVec[index] = value;
}
//表示矩阵逐个元素相乘
template<typename DataType>
Vector<DataType> Vector<DataType>::mul(Vector<DataType>& mat) {
	if (mat.GetLength() != this->GetLength()) {
		throw "The size don't match!";
	}
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType value = this->InnerVec[k] * mat.Get(k);
		resultMat.Set(value, k);
	}
	return resultMat;
}
template<typename DataType>
Vector<DataType> Vector<DataType>::mul(DataType value) {
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType val = this->InnerVec[k] * value;
		resultMat.Set(val, k);
	}
	return resultMat;
}

//表示矩阵逐个元素相加
template<typename DataType>
Vector<DataType> Vector<DataType>::add(Vector<DataType>& mat) {
	if (mat.GetLength() != this->GetLength()) {
		throw "The size don't match!";
	}
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType value = this->InnerVec[k] + mat.Get(k);
		resultMat.Set(value, k);
	}
	return resultMat;
}
template<typename DataType>
Vector<DataType> Vector<DataType>::add(DataType value) {
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType val = this->InnerVec[k] + value;
		resultMat.Set(val, k);
	}
	return resultMat;
}
//表示矩阵中逐个元素相减
template<typename DataType>
Vector<DataType> Vector<DataType>::sub(Vector<DataType>& mat) {
	if (mat.GetLength() != this->GetLength()) {
		throw "The size don't match!";
	}
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType value = this->InnerVec[k] - mat.Get(k);
		resultMat.Set(value, k);
	}
	return resultMat;
}
template<typename DataType>
Vector<DataType> Vector<DataType>::sub(DataType value) {
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType val = this->InnerVec[k] - value;
		resultMat.Set(val, k);
	}
	return resultMat;
}
//表示矩阵中逐个元素相除
template<typename DataType>
Vector<DataType> Vector<DataType>::div(Vector<DataType>& mat) {
	if (mat.GetLength() != this->GetLength()) {
		throw "The size don't match!";
	}
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType value = this->InnerVec[k] / mat.Get(k);
		resultMat.Set(value, k);
	}
	return resultMat;
}
template<typename DataType>
Vector<DataType> Vector<DataType>::div(DataType value) {
	Vector<DataType> resultMat(this->Length);
	for (unsigned int k = 0; k < this->Length; k++) {
		DataType val = this->InnerVec[k] / value;
		resultMat.Set(val, k);
	}
	return resultMat;
}
//Vector 对Matrix广播相加或者相减

template<typename DataType>
Matrix<DataType> Vector<DataType>::sub(Matrix<DataType>& mat) {
	if (mat.GetColumnLength() != this->Length) {
		throw "The size don't match!";
	}
	Matrix<DataType> Result(mat.size());
	for (unsigned int k = 0; k<this->mat.GetRowLength(); k++) {
		for (unsigned int j = 0; j<this->mat.GetColumnLength(); j++) {
			Result.Set(this->InnerMat[j] - mat.Get(k, j), k, j);
		}
	}
	return Result;
}
template<typename DataType>
Matrix<DataType> Vector<DataType>::add(Matrix<DataType>& mat) {
	if (mat.GetColumnLength() != this->Length) {
		throw "The size don't match!";
	}
	Matrix<DataType> Result(mat.size());
	for (unsigned int k = 0; k<this->mat.GetRowLength(); k++) {
		for (unsigned int j = 0; j<this->mat.GetColumnLength(); j++) {
			Result.Set(this->InnerMat[j] + mat.Get(k, j), k, j);
		}
	}
	return Result;
}
//将矩阵中的元素转化为字符串形式
template<typename DataType>
std::string Vector<DataType>::toString() {
	unsigned int Length = this->Length;
	std::stringstream tmp_stream;
	tmp_stream.precision(4);  //由于未用定点格式，这里设置的是保留4位有效数字
	std::string str_size;
	tmp_stream << "Size:[" << Length << "]";
	tmp_stream >> str_size;
	str_size = str_size + "\n";
	//注意用完s_stream要清除字符流
	tmp_stream.clear();
	std::string str_mat = "";
	for (unsigned int k = 0; k<Length; k++) {
		tmp_stream << this->InnerVec[k];
		std::string line;
		tmp_stream >> line;
		str_mat = str_mat + line + "\t";
		tmp_stream.clear();
	}
	str_mat = str_mat + "\n";
	return str_mat + str_size;
}
#endif // !VECTOR_H


