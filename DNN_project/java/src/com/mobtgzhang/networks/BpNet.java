package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
import com.mobtgzhang.utils.Activate;
import com.mobtgzhang.utils.MathTool;
import com.mobtgzhang.utils.activates.Relu;
import com.mobtgzhang.utils.activates.Sigmoid;
import com.mobtgzhang.utils.activates.SoftPlus;
import com.mobtgzhang.utils.activates.Tanh;

public class BpNet {
    private Matrix weightIn;
    private Vector baisIn;
    private Matrix weightOut;
    private Vector baisOut;
    private Activate actFunc;
    private String act_name;
    private int in_dim;
    private int hid_dim;
    private int out_dim;
    private Activate selectAct(String act_name) throws Exception {
        switch (act_name){
            case "Sigmoid":
                return new Sigmoid();
            case "Relu":
                return new Relu();
            case "Tanh":
                return new Tanh();
            case "SoftPlus":
                return new SoftPlus();
            default:
                throw new Exception("Unknown activate function: "+act_name+
                        "and add it to \"class Activate.\"");
        }
    }
    public BpNet(int in_dim,int hid_dim,int out_dim,String act_name) throws Exception {
        this.in_dim = in_dim;
        this.hid_dim = hid_dim;
        this.out_dim = out_dim;
        this.act_name = act_name;
        this.weightIn = new Matrix(in_dim,hid_dim);
        this.baisIn = new Vector(hid_dim);
        this.actFunc = this.selectAct(act_name);
        this.weightOut = new Matrix(hid_dim,out_dim);
        this.baisOut = new Vector(out_dim);
    }
    public BpNet(int in_dim,int hid_dim,int out_dim) throws Exception {
        this(in_dim,hid_dim,out_dim,"Sigmoid");
    }
    public Matrix forward(Matrix input) throws Exception {
        Matrix hid = MathTool.add(MathTool.dot(input,this.weightIn),this.baisIn);
        Matrix act_hid = this.actFunc.forward(hid);
        Matrix out = MathTool.add(MathTool.dot(act_hid,this.weightOut),this.baisOut);
        return out;
    }
    public Vector forward(Vector input) throws Exception {
        Vector hid = MathTool.add(MathTool.dot(input,this.weightIn),this.baisIn);
        Vector act_hid = this.actFunc.forward(hid);
        Vector out = MathTool.add(MathTool.dot(act_hid,this.weightOut),this.baisOut);
        return out;
    }
}
