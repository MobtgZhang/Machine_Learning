package utils

import (
	"DNNs/matrix"
	"fmt"
)

func Linspace(down ,up float64,length uint)*matrix.Vector{
	returnVec :=matrix.NewVector(length)
	delta := (up-down)/(float64)(length-1)
	var k uint
	for k=0;k<length;k++{
		value := delta*(float64)(k)
		returnVec.Set(value,k)
	}
	return returnVec
}
func MatDot(MatA,MatB *matrix.Matrix)*matrix.Matrix{
	if(MatA.GetColumn()!=MatB.GetRow()){
		message := fmt.Sprintf("The two matrix size: MatrixA:(%d,%d),MatrixB:(%d,%d) don't match!",
			MatA.GetRow(),MatA.GetColumn(),MatB.GetRow(),MatB.GetColumn())
		panic(message)
	}
	rowlength := MatA.GetRow()
	columlength :=MatB.GetColumn()
	tmplength :=MatA.GetColumn()
	returnMat :=matrix.NewMatrix(rowlength,columlength)
	var k,j,i uint
	for k=0;k<rowlength;k++{
		for j=0;j<columlength;j++{
			var value float64 = 0.0
			for i=0;i<tmplength;i++{
				value+=MatA.Get(k,i)*MatB.Get(i,j)
			}
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
func VecDot(Vec *matrix.Vector,Mat *matrix.Matrix)*matrix.Vector{
	if(Vec.Size()!=Mat.GetRow()){
		message := fmt.Sprintf("The two matrix size: Vector:(%d,),Matrix:(%d,%d) don't match!",
			Vec.Size(),Mat.GetRow(),Mat.GetColumn())
		panic(message)
	}
	length :=Mat.GetColumn()
	tmplen :=Mat.GetRow()
	returnVec :=matrix.NewVector(length)
	var k,j uint
	for k=0;k<length;k++{
		var value float64= 0
		for j=0;j<tmplen;j++{
			value+= Vec.Get(j)*Mat.Get(j,k)
		}
		returnVec.Set(value,k)
	}
	return returnVec
}
func SelectActMat(mat *matrix.Matrix,act_name string)*matrix.Matrix{
	switch act_name {
	case "Sigmoid":
		return NewSigmoid().ForwardMat(mat)
	case "Relu":
		return NewRelu(0).ForwardMat(mat)
	case "Tanh":
		return NewTanh().ForwardMat(mat)
	case "SoftPlus":
		return NewSoftPlus().ForwardMat(mat)
	default:
		message :=fmt.Sprintf("Unknown activate function: %s",act_name)
		panic(message)
	}
}
func SelectActVec(vec *matrix.Vector,act_name string)*matrix.Vector{
	switch act_name {
	case "Sigmoid":
		return NewSigmoid().ForwardVec(vec)
	case "Relu":
		return NewRelu(0).ForwardVec(vec)
	case "Tanh":
		return NewTanh().ForwardVec(vec)
	case "SoftPlus":
		return NewSoftPlus().ForwardVec(vec)
	default:
		message :=fmt.Sprintf("Unknown activate function: %s",act_name)
		panic(message)
	}
}