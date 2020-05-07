package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
import com.mobtgzhang.utils.Activate;
import com.mobtgzhang.utils.MathTool;

public abstract class BpNet {
    protected Matrix weightIn;
    protected Vector baisIn;
    protected Matrix weightOut;
    protected Vector baisOut;
    protected Activate actFunc;
    protected String act_name;
    protected int in_dim;
    protected int hid_dim;
    protected int out_dim;
    public BpNet(int in_dim,int hid_dim,int out_dim,String act_name) throws Exception {
        this.in_dim = in_dim;
        this.hid_dim = hid_dim;
        this.out_dim = out_dim;
        this.act_name = act_name;
        this.weightIn = new Matrix(in_dim,hid_dim);
        this.baisIn = new Vector(hid_dim);
        this.actFunc = MathTool.selectAct(act_name);
        this.weightOut = new Matrix(hid_dim,out_dim);
        this.baisOut = new Vector(out_dim);
    }
    public BpNet(int in_dim,int hid_dim,int out_dim) throws Exception {
        this(in_dim,hid_dim,out_dim,"Sigmoid");
    }
    protected Matrix forward(Matrix input) throws Exception {
        Matrix hid = MathTool.add(MathTool.dot(input,this.weightIn),this.baisIn);
        Matrix act_hid = this.actFunc.forward(hid);
        Matrix out = MathTool.add(MathTool.dot(act_hid,this.weightOut),this.baisOut);
        return out;
    }
    protected Vector forward(Vector input) throws Exception {
        Vector hid = MathTool.add(MathTool.dot(input,this.weightIn),this.baisIn);
        Vector act_hid = this.actFunc.forward(hid);
        Vector out = MathTool.add(MathTool.dot(act_hid,this.weightOut),this.baisOut);
        return out;
    }
    public abstract void backward();
}
