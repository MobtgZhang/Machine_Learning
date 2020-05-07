package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
import com.mobtgzhang.utils.activates.SoftMax;

public class BpNetClassification extends BpNet{
    private SoftMax predict;
    public BpNetClassification(int in_dim,int hid_dim,int out_dim,String act_name) throws Exception {
        super(in_dim,hid_dim,out_dim,act_name);
        this.predict = new SoftMax();
    }
    public BpNetClassification(int in_dim,int hid_dim,int out_dim) throws Exception {
        this(in_dim,hid_dim,out_dim,"Sigmoid");
    }
    public Matrix forward(Matrix input) throws Exception {
        Matrix out = super.forward(input);
        return this.predict.forward(out);
    }
    public Vector forward(Vector input) throws Exception {
        Vector out = super.forward(input);
        return this.predict.forward(out);
    }
    @Override
    public void backward() {

    }
}
