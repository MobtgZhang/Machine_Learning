#pragma once
#ifndef MATRIX_H
#define MATRIX_H
struct Size{
	unsigned int RowLength;
	unsigned int ColumnLength;
	Size(unsigned int r,unsigned int c){
		this->RowLength = r;
		this->ColumnLength = c;
	}
	Size(Size&size){
		this->RowLength = size.RowLength;
		this->ColumnLength = size.ColumnLength;
	}
};
template<typename DataType>
class Matrix {
private:
	Size mat_size;
	DataType** InnerMat;
public:
	//构造函数
	Matrix(Matrix<DataType>&mat);
	Matrix(unsigned int RowLength, unsigned int ColumnLength);
	Matrix(Size size);
	//获取矩阵的大小
	unsigned int GetRowLength() { return this->mat_size.RowLength; }
	unsigned int GetColumnLength() { return this->mat_size.ColumnLength; }
	Size size(){return this->mat_size;}
	//表示矩阵逐个元素相乘
	Matrix<DataType> mul(Matrix<DataType>& mat);
	Matrix<DataType> mul(DataType value);
	//获取矩阵中的某一个元素
	DataType Get(unsigned int rowindex,unsigned int columninex);
	//设置矩阵中的某一个元素
	void Set(DataType value,unsigned int rowindex,unsigned int columninex);
	//表示矩阵逐个元素相加
	Matrix<DataType> add(Matrix<DataType>& mat);
	Matrix<DataType> add(DataType value);
	//表示矩阵中逐个元素相减
	Matrix<DataType> sub(Matrix<DataType>& mat);
	Matrix<DataType> sub(DataType value);
	//表示矩阵中逐个元素相除
	Matrix<DataType> div(Matrix<DataType>& mat);
	Matrix<DataType> div(DataType value);
	//重载运算符
	Matrix<DataType> operator +(Matrix<DataType>& mat) { return this->add(mat); }
	Matrix<DataType> operator +(DataType value) { return this->add(value); }
	Matrix<DataType> operator -(Matrix<DataType>& mat) { return this->sub(mat); }
	Matrix<DataType> operator -(DataType value) { return this->sub(value); }
	Matrix<DataType> operator *(Matrix<DataType>& mat) { return this->mul(mat); }
	Matrix<DataType> operator *(DataType value) { return this->mul(value); }
	Matrix<DataType> operator /(Matrix<DataType>& mat) { return this->div(mat); }
	Matrix<DataType> operator /(DataType value) { return this->div(value); }
	//析构函数
	~Matrix();
};
//构造函数的实现
template<typename DataType>
Matrix<DataType>::Matrix(unsigned int RowLength, unsigned int ColumnLength) {
	this->mat_size.RowLength = RowLength;
	this->mat_size.ColumnLength = ColumnLength;
	this->InnerMat = new DataType[RowLength][ColumnLength];
}
template<typename DataType>
Matrix<DataType>::Matrix(Size& size){
	this->mat_size.RowLength = size.RowLength;
	this->mat_size.ColumnLength = size.ColumnLength;
	this->InnerMat = new DataType[size.RowLength][size.ColumnLength];
}
template<typename DataType>
Matrix<DataType>::Matrix(Matrix<DataType>&mat) {
	this->mat_size = Size(mat.size())
	this->InnerMat = new DataType[this.mat_size.RowLength][this.mat_size.ColumnLength];
}
//析构函数的实现
template<typename DataType>
Matrix<DataType>::~Matrix() {
	delete []this->InnerMat;
}
template<typename DataType>
Matrix<DataType>::add(Matrix<DataType>& mat) {
	if (mat.GetRowLength() != this->mat_size.RowLength && mat.GetColumnLength() != this->mat_size.ColumnLength) {
		throw "The size of two matrix don't match!"
	}
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for(unsigned int k=0;k<this->mat_size.RowLength;k++){
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Rsult.Set(this->InnerMat[k,j]+mat.Get(k,j),k,j);
		}
	}
	return Rsult;
}
template<typename DataType>
Matrix<DataType>::add(DataType value) {
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for (unsigned int k = 0; k < this->mat_size.RowLength; k++) {
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Result.Set(this->InnerMat[k][j]+value,k,j); 
		}
	}
	return Rsult;
}
template<typename DataType>
Matrix<DataType>::sub(Matrix<DataType>& mat) {
	if (mat.GetRowLength() != this->mat_size.RowLength && mat.GetColumnLength() != this->mat_size.ColumnLength) {
		throw "The size of two matrix don't match!"
	}
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for(unsigned int k=0;k<this->mat_size.RowLength;k++){
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Rsult.Set(this->InnerMat[k,j]-mat.Get(k,j),k,j);
		}
	}
	return Rsult;
}
template<typename DataType>
Matrix<DataType>::sub(DataType value) {
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for (unsigned int k = 0; k < this->mat_size.RowLength; k++) {
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Result.Set(this->InnerMat[k][j]-value,k,j); 
		}
	}
	return Rsult;
}
template<typename DataType>
Matrix<DataType>::div(Matrix<DataType>& mat) {
	if (mat.GetRowLength() != this->mat_size.RowLength && mat.GetColumnLength() != this->mat_size.ColumnLength) {
		throw "The size of two matrix don't match!"
	}
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for(unsigned int k=0;k<this->mat_size.RowLength;k++){
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Rsult.Set(this->InnerMat[k,j]/mat.Get(k,j),k,j);
		}
	}
	return Rsult;
}
template<typename DataType>
Matrix<DataType>::div(DataType value) {
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for (unsigned int k = 0; k < this->mat_size.RowLength; k++) {
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Result.Set(this->InnerMat[k][j]/value,k,j); 
		}
	}
	return Rsult;
}

template<typename DataType>
Matrix<DataType>::mul(Matrix<DataType>& mat) {
	if (mat.GetRowLength() != this->mat_size.RowLength && mat.GetColumnLength() != this->mat_size.ColumnLength) {
		throw "The size of two matrix don't match!"
	}
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for(unsigned int k=0;k<this->mat_size.RowLength;k++){
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Rsult.Set(this->InnerMat[k,j]*mat.Get(k,j),k,j);
		}
	}
	return Rsult;
}
template<typename DataType>
Matrix<DataType>::mul(DataType value) {
	Matrix<DataType> Result(this->mat_size.RowLength,this->mat_size.ColumnLength);
	for (unsigned int k = 0; k < this->mat_size.RowLength; k++) {
		for(unsigned int j=0;j<this->mat_size.ColumnLength;j++){
			Result.Set(this->InnerMat[k][j]*value,k,j); 
		}
	}
	return Rsult;
}
#endif
