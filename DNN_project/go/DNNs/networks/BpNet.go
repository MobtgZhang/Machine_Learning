package networks

import "DNNs/matrix"
import "DNNs/utils"

type BpNet struct {
	in_dim uint
	hid_dim uint
	out_dim uint
	act_name string
	weightIn *matrix.Matrix
	baisIn *matrix.Vector
	weightOut *matrix.Matrix
	baisOut *matrix.Vector
}

func NewBpNet(in_dim,hid_dim,out_dim uint,act_name string)*BpNet{
	weightIn := matrix.NewMatrix(in_dim,hid_dim)
	baisIn := matrix.NewVector(hid_dim)
	weightOut :=matrix.NewMatrix(hid_dim,out_dim)
	baisOut := matrix.NewVector(out_dim)
	return&BpNet{in_dim,hid_dim,out_dim,act_name,
		weightIn,baisIn,weightOut,baisOut}
}
func (this*BpNet)ForwardMat(input *matrix.Matrix)*matrix.Matrix{
	hid := utils.MatDot(input,this.weightIn).AddVector(this.baisIn)
	hid_act := utils.SelectActMat(hid,this.act_name)
	output := utils.MatDot(hid_act,this.weightOut).AddVector(this.baisOut)
	return output
}
func (this*BpNet)ForwardVec(input *matrix.Vector)*matrix.Vector{
	hid := utils.VecDot(input,this.weightIn).Add(this.baisIn)
	hid_act := utils.SelectActVec(hid,this.act_name)
	output := utils.VecDot(hid_act,this.weightOut).Add(this.baisOut)
	return output
}
