package utils

import (
	"DNNs/matrix"
	"math"
)
type ActInterface interface {
	ForwardMat(mat *matrix.Matrix)*matrix.Matrix
	ForwardVec(vec *matrix.Vector)*matrix.Vector
	InnerFunc(value float64)float64
}
type Activate struct {
	act_name string
}
//Sigmoid函数
type Sigmoid struct {
	Activate
}
func NewSigmoid()*Sigmoid{
	return&Sigmoid{Activate{"Sigmoid"}}
}

func (this*Sigmoid)ForwardMat(mat *matrix.Matrix)*matrix.Matrix{
	rowlength,columnlength :=mat.Size()
	returnMat :=matrix.NewMatrix(rowlength,columnlength)
	var k,j uint
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := 1.0/(1.0+math.Exp(mat.Get(k,j)))
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
func (this*Sigmoid)ForwardVec(vec *matrix.Vector)*matrix.Vector{
	length :=vec.Size()
	returnVec := matrix.NewVector(length)
	var k uint
	for k=0;k<length;k++{
		value := 1.0/(1.0+math.Exp(vec.Get(k)))
		returnVec.Set(value,k)
	}
	return returnVec
}
//Relu函数
type Relu struct {
	Activate
	alpha float64
}
func NewRelu(alpha float64)*Relu{
	return&Relu{Activate{"Relu"},alpha}
}

func (this*Relu)ForwardMat(mat *matrix.Matrix)*matrix.Matrix{
	rowlength,columnlength :=mat.Size()
	returnMat :=matrix.NewMatrix(rowlength,columnlength)
	var k,j uint
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			var value float64
			if mat.Get(k,j) >0 {
				value = mat.Get(k,j)
			}else{
				value = mat.Get(k,j)*this.alpha
			}
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
func (this*Relu)ForwardVec(vec *matrix.Vector)*matrix.Vector{
	length :=vec.Size()
	returnVec :=matrix.NewVector(length)
	var k uint
	for k=0;k<length;k++{
		var value float64
		if vec.Get(k) >0 {
			value = vec.Get(k)
		}else{
			value = vec.Get(k)*this.alpha
		}
		returnVec.Set(value,k)
	}
	return returnVec
}
//SoftPlus函数
type SoftPlus struct {
	Activate
}
func NewSoftPlus()*SoftPlus{
	return&SoftPlus{Activate{"SoftPlus"}}
}

func (this*SoftPlus)ForwardMat(mat *matrix.Matrix)*matrix.Matrix{
	rowlength,columnlength :=mat.Size()
	returnMat :=matrix.NewMatrix(rowlength,columnlength)
	var k,j uint
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := math.Log(1+math.Exp(mat.Get(k,j)))
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
func (this*SoftPlus)ForwardVec(vec *matrix.Vector)*matrix.Vector{
	length :=vec.Size()
	returnVec :=matrix.NewVector(length)
	var k uint
	for k=0;k<length;k++{
		value := math.Log(1+math.Exp(vec.Get(k)))
		returnVec.Set(value,k)
	}
	return returnVec
}

//Tanh函数
type Tanh struct {
	Activate
}

func NewTanh()*Tanh{
	return&Tanh{Activate{"Tanh"}}
}

func (this*Tanh)ForwardMat(mat *matrix.Matrix)*matrix.Matrix{
	rowlength,columnlength :=mat.Size()
	returnMat :=matrix.NewMatrix(rowlength,columnlength)
	var k,j uint
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := math.Tanh(mat.Get(k,j))
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
func (this*Tanh)ForwardVec(vec *matrix.Vector)*matrix.Vector{
	length :=vec.Size()
	returnVec :=matrix.NewVector(length)
	var k uint
	for k=0;k<length;k++{
		value := math.Tanh(vec.Get(k))
		returnVec.Set(value,k)
	}
	return returnVec
}
