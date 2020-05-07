package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
import com.mobtgzhang.utils.MathTool;

public class Linear {
    private Matrix weight;
    private Vector bais;
    public Linear(int in_dim,int out_dim){
        this.weight = new Matrix(in_dim,out_dim);
        this.bais = new Vector(out_dim);
    }
    public Matrix forward(Matrix input) throws Exception {
        return MathTool.add(MathTool.dot(input,this.weight),this.bais);
    }
    public Vector forward(Vector input) throws Exception {
        return MathTool.add(MathTool.dot(input,this.weight),this.bais);
    }
}
