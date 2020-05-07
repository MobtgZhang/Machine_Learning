package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
public class BpNetRgression extends BpNet{
    public BpNetRgression(int in_dim,int hid_dim,int out_dim,String act_name) throws Exception {
        super(in_dim,hid_dim,out_dim,act_name);
    }
    public BpNetRgression(int in_dim,int hid_dim,int out_dim) throws Exception {
        super(in_dim,hid_dim,out_dim,"Sigmoid");
    }
    public Matrix forward(Matrix input) throws Exception {
        return super.forward(input);
    }
    public Vector forward(Vector input) throws Exception {
        return super.forward(input);
    }
    public void backward(){

    }
}
