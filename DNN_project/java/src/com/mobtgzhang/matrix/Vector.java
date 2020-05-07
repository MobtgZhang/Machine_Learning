package com.mobtgzhang.matrix;

import java.text.DecimalFormat;
import java.util.Random;

public class Vector {
    private int length;
    private double[] vector;
    public Vector(int length,boolean rand){
        this.length = length;
        this.vector = new double[length];
        for(int k=0;k<length;k++){
            if (rand){
                Random tmpRand = new Random();
                this.vector[k] = tmpRand.nextDouble();
            }else{
                for(int j=0;j<length;j++){
                    this.vector[k] = 0.0;
                }
            }
        }
    }
    public Vector(int length){
        this(length,true);
    }

    public int getLength() {
        return length;
    }
    public double get(int index){
        return this.vector[index];
    }
    public void set(double value,int index){
        this.vector[index]=value;
    }
    @Override
    public String toString() {
        String str_size = "Size:["+this.length+"]\n";
        String str_vec = "";
        DecimalFormat df = new DecimalFormat("0.0000");
        for(int k=0;k<this.length;k++){
                str_vec += df.format(this.vector[k]);
            str_vec += "\t";
            }
        str_vec+="\n";
        return str_vec + str_size;
    }

    @Override
    public Vector clone() throws CloneNotSupportedException {
        Vector returnVec = new Vector(this.length,false);
        for(int k=0;k<this.length;k++){
            returnVec.set(this.vector[k],k);
        }
        return returnVec;
    }
}
