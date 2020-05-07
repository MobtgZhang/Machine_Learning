package matrix

import "fmt"

//定义向量
type Vector struct {
	length uint
	vector [] float64
}
//构造函数
func NewVector(length uint)*Vector{
	data_mat := make([]float64,length)
	return&Vector{length,data_mat}
}
//获取矩阵元素
func (this*Vector)Get(index uint) float64{
	return this.vector[index]
}
//设置矩阵中的元素
func (this*Vector)Set(value float64,index uint){
	this.vector[index] = value
}
//获取矩阵的大小
func (this*Vector)Size() uint{
	return this.length
}
//将两个向量相加
func (this*Vector)Add(Mat *Vector)*Vector{
	if (Mat.Size()!=this.Size()){
		message := fmt.Sprintf("The two matrix size: VectorA:(%d),VectorB:(%d) don't match!",
			Mat.Size(),Mat.Size())
		panic(message)
	}
	length :=Mat.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]+Mat.Get(k),k)
	}
	return returnMat
}
//将两个向量相减
func (this*Vector)Sub(Mat *Vector)*Vector{
	if (Mat.Size()!=this.Size()){
		message := fmt.Sprintf("The two matrix size: VectorA:(%d),VectorB:(%d) don't match!",
			Mat.Size(),Mat.Size())
		panic(message)
	}
	length :=Mat.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]-Mat.Get(k),k)
	}
	return returnMat
}
//将两个向量相乘
func (this*Vector)Mul(Mat *Vector)*Vector{
	if (Mat.Size()!=this.Size()){
		message := fmt.Sprintf("The two matrix size: VectorA:(%d),VectorB:(%d) don't match!",
			Mat.Size(),Mat.Size())
		panic(message)
	}
	length :=Mat.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]*Mat.Get(k),k)
	}
	return returnMat
}
//将两个向量相除
func (this*Vector)Div(Mat *Vector)*Vector{
	if (Mat.Size()!=this.Size()){
		message := fmt.Sprintf("The two matrix size: VectorA:(%d),VectorB:(%d) don't match!",
			Mat.Size(),Mat.Size())
		panic(message)
	}
	length :=Mat.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]+Mat.Get(k),k)
	}
	return returnMat
}
//将值与向量相加
func (this*Vector)AddValue(value float64)*Vector{
	length :=this.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]+value,k)
	}
	return returnMat
}
//将值与向量相减
func (this*Vector)SubValue(value float64)*Vector{
	length :=this.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]-value,k)
	}
	return returnMat
}
//将值与向量相乘
func (this*Vector)MulValue(value float64)*Vector{
	length :=this.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]*value,k)
	}
	return returnMat
}
//将值与向量相乘
func (this*Vector)DivValue(value float64)*Vector{
	length :=this.Size()
	var k uint
	returnMat := NewVector(length)
	for k=0;k<length;k++{
		returnMat.Set(this.vector[k]/value,k)
	}
	return returnMat
}
