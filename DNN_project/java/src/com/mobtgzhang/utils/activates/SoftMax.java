package com.mobtgzhang.utils.activates;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;

public class SoftMax {
    private String act_name;
    public SoftMax(){
        this.act_name = "SoftMax";
    }
    public Vector forward(Vector input){
        Vector resultMat = new Vector(input.getLength(),false);
        double sum_val = 0;
        for(int k=0;k<input.getLength();k++){
            sum_val += Math.exp(input.get(k));
        }
        for(int k=0;k<input.getLength();k++){
            resultMat.set(Math.exp(input.get(k))/sum_val,k);
        }
        return resultMat;
    }
    public Matrix forward(Matrix input){
        int batch = input.getRowlength();
        int length = input.getColumnlength();
        Matrix resultMat = new Matrix(batch,length,false);
        for(int k=0;k<batch;k++){
            double sum_val = 0;
            for(int j=0;j<length;j++){
                sum_val += Math.exp(input.get(k,j));
            }
            for(int j=0;j<length;j++){
                resultMat.set(Math.exp(input.get(k,j))/sum_val,k,j);
            }
        }
        return resultMat;
    }
}
