package matrix

import "fmt"

//定义矩阵
type Matrix struct {
	RowLength uint
	ColumnLength uint
	data_mat [][]float64
}
//初始化矩阵，构造函数
func NewMatrix(RowLength,ColumnLength uint)*Matrix{
	data_mat := make([][]float64,RowLength)
	for k:=0;k<len(data_mat);k++{
		data_mat[k] = make([]float64,ColumnLength)
	}
	return&Matrix{RowLength,ColumnLength,data_mat}
}
//获取矩阵元素
func (this*Matrix)Get(rowindex uint,columnindex uint) float64{
	return this.data_mat[rowindex][columnindex]
}
//设置矩阵中的元素
func (this*Matrix)Set(value float64,rowindex uint,columnindex uint){
	this.data_mat[rowindex][columnindex] = value
}
//获取矩阵的大小
func (this*Matrix)Size() (uint,uint){
	return this.RowLength,this.ColumnLength
}
func (this*Matrix)GetRow()uint{
	return this.RowLength
}
func (this*Matrix)GetColumn()uint{
	return this.ColumnLength
}
//将两个矩阵相加
func (this*Matrix)Add(Mat *Matrix)*Matrix{
	if (Mat.GetRow()!=this.GetRow() && Mat.GetColumn()!=this.GetColumn()){
		message := fmt.Sprintf("The two matrix size: MatrixA:(%d,%d),MatrixB:(%d,%d) don't match!",
			Mat.GetRow(),Mat.GetColumn(),this.GetRow(),this.GetColumn())
		panic(message)
	}
	rowlength,columnlength :=Mat.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := Mat.Get(k,j)+this.Get(k,j)
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将两个矩阵相减
func (this*Matrix)Sub(Mat *Matrix)*Matrix{
	if (Mat.GetRow()!=this.GetRow() && Mat.GetColumn()!=this.GetColumn()){
		message := fmt.Sprintf("The two matrix size: MatrixA:(%d,%d),MatrixB:(%d,%d) don't match!",
			Mat.GetRow(),Mat.GetColumn(),this.GetRow(),this.GetColumn())
		panic(message)
	}
	rowlength,columnlength :=Mat.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := Mat.Get(k,j)-this.Get(k,j)
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将两个矩阵相乘
func (this*Matrix)Mul(Mat *Matrix)*Matrix{
	if (Mat.GetRow()!=this.GetRow() && Mat.GetColumn()!=this.GetColumn()){
		message := fmt.Sprintf("The two matrix size: MatrixA:(%d,%d),MatrixB:(%d,%d) don't match!",
			Mat.GetRow(),Mat.GetColumn(),this.GetRow(),this.GetColumn())
		panic(message)
	}
	rowlength,columnlength :=Mat.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := Mat.Get(k,j)*this.Get(k,j)
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将两个矩阵相除
func (this*Matrix)Div(Mat *Matrix)*Matrix{
	if (Mat.GetRow()!=this.GetRow() && Mat.GetColumn()!=this.GetColumn()){
		message := fmt.Sprintf("The two matrix size: MatrixA:(%d,%d),MatrixB:(%d,%d) don't match!",
			Mat.GetRow(),Mat.GetColumn(),this.GetRow(),this.GetColumn())
		panic(message)
	}
	rowlength,columnlength :=Mat.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := Mat.Get(k,j)/this.Get(k,j)
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将矩阵与值相加
func (this*Matrix)AddValue(value float64)*Matrix{
	rowlength,columnlength :=this.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := this.Get(k,j)/value
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将矩阵与值相减
func (this*Matrix)SubValue(value float64)*Matrix{
	rowlength,columnlength :=this.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := this.Get(k,j)/value
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将矩阵与值相乘
func (this*Matrix)MulValue(value float64)*Matrix{
	rowlength,columnlength :=this.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := this.Get(k,j)/value
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//将矩阵与值相除
func (this*Matrix)DivValue(value float64)*Matrix{
	rowlength,columnlength :=this.Size()
	var k,j uint
	returnMat := NewMatrix(rowlength,columnlength)
	for k=0;k<rowlength;k++{
		for j=0;j<columnlength;j++{
			value := this.Get(k,j)/value
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
//与向量相加或者相减
func(this*Matrix)AddVector(vec *Vector)*Matrix{
	if (this.GetColumn()!=vec.Size()){
		message := fmt.Sprintf("The two matrix size: Matrix:(%d,%d),Vector:(%d) don't match!",
			this.GetRow(),this.GetColumn(),vec.Size())
		panic(message)
	}
	var k,j uint
	returnMat := NewMatrix(this.RowLength,this.ColumnLength)
	for k=0;k<this.RowLength;k++{
		for j=0;j<this.ColumnLength;j++{
			value := this.Get(k,j)+vec.Get(j)
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}
func(this*Matrix)SubVector(vec *Vector)*Matrix{
	if (this.GetColumn()!=vec.Size()){
		message := fmt.Sprintf("The two matrix size: Matrix:(%d,%d),Vector:(%d) don't match!",
			this.GetRow(),this.GetColumn(),vec.Size())
		panic(message)
	}
	var k,j uint
	returnMat := NewMatrix(this.RowLength,this.ColumnLength)
	for k=0;k<this.RowLength;k++{
		for j=0;j<this.ColumnLength;j++{
			value := this.Get(k,j)-vec.Get(j)
			returnMat.Set(value,k,j)
		}
	}
	return returnMat
}

