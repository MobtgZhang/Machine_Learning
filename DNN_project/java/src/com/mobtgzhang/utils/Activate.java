package com.mobtgzhang.utils;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;

public abstract class Activate {
    private String act_name;
    public Activate(String act_name){
        this.act_name = act_name;
    }

    public String getAct_name() {
        return act_name;
    }
    public abstract double innerFunc(double value);
    public Matrix forward(Matrix input){
        int rowlength = input.getRowlength();
        int columnlength = input.getColumnlength();
        Matrix returnMat = new Matrix(input.getRowlength(),input.getColumnlength(),false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(this.innerFunc(input.get(k,j)),k,j);
            }
        }
        return returnMat;
    }
    public Vector forward(Vector input){
        int length = input.getLength();
        Vector returnVec = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnVec.set(this.innerFunc(input.get(k)),k);
        }
        return returnVec;
    }
}

