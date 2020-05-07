package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;

public class DNNRegression extends DNNNet{
    public DNNRegression(int[] dim_list, String act_name) throws Exception {
        super(dim_list, act_name);
    }
    public DNNRegression(int[] dim_list) throws Exception {
        this(dim_list, "Sigmoid");
    }
    public Matrix forward(Matrix input) throws Exception {
        return super.forward(input);
    }
    public Vector forward(Vector input) throws Exception {
        return super.forward(input);
    }
    @Override
    public void backward() {

    }
}
