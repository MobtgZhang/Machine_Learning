package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
import com.mobtgzhang.utils.activates.SoftMax;

public class DNNClassification extends DNNNet{
    private SoftMax predict;
    public DNNClassification(int[] dim_list, String act_name) throws Exception {
        super(dim_list, act_name);
        this.predict = new SoftMax();
    }
    public DNNClassification(int[] dim_list) throws Exception {
        this(dim_list, "Sigmoid");
    }
    public Matrix forward(Matrix input) throws Exception {
        Matrix output = super.forward(input);
        return this.predict.forward(output);
    }
    public Vector forward(Vector input) throws Exception {
        Vector output = super.forward(input);
        return this.predict.forward(output);
    }
    @Override
    public void backward() {

    }
}
