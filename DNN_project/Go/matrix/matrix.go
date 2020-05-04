package matrix

import "fmt"

type Object interface {
	add(a ,b Object) Object
	sub(a ,b Object) Object
	mul(a ,b Object) Object
	div(a ,b Object) Object
}
//定义矩阵
type Matrix struct {
	RowLength uint
	ColumnLength uint
	data_mat [][]Object
}
//初始化矩阵，构造函数
func NewMatrix(RowLength,ColumnLength uint)*Matrix{
	var data_mat []([]Object) = make([]([]Object),RowLength)
	for k:=0;k<len(data_mat);k++{
		data_mat[k] = make([]Object,ColumnLength)
	}
	return&Matrix{RowLength,ColumnLength,data_mat}
}
//获取矩阵元素
func (this*Matrix)Get(rowindex uint,columnindex uint) interface{}{
	return this.data_mat[rowindex][columnindex]
}
//设置矩阵中的元素
func (this*Matrix)Set(object Object,rowindex uint,columnindex uint){
	this.data_mat[rowindex][columnindex] = object
}
//获取矩阵的大小
func (this*Matrix)Size() (uint,uint){
	return this.RowLength,this.ColumnLength
}
//将两个矩阵相加
func (this*Matrix)Add(Mat *Matrix)*Matrix{
	rlen,clen :=Mat.Size()
	if(rlen!=this.RowLength && clen!=this.ColumnLength){
		panic(fmt.Sprintf("The size:(%d,%d) doesn't match the size(%d,%d)",
			rlen,clen,this.RowLength,this.ColumnLength))
	}
	resultMat := NewMatrix(rlen,clen)
	var k,j uint
	for k=0;k<rlen;k++{
		for j=0;j<clen;j++{
			result := Object.add(this.Get(k,j),Mat.Get(k,j))
			resultMat.Set(result,k,j)
		}
	}
	return resultMat
}