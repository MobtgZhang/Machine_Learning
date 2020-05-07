package com.mobtgzhang.matrix;

import java.text.DecimalFormat;
import java.util.Random;

public class Matrix {
    private int rowlength;
    private int columnlength;
    private double[][] matrix;
    public Matrix(int rowlength,int columnlength,boolean rand){
        this.rowlength = rowlength;
        this.columnlength = columnlength;
        this.matrix = new double[rowlength][];
        for(int k=0;k<rowlength;k++){
            this.matrix[k] = new double[columnlength];
            if (rand){
                Random tmpRand = new Random();
                for(int j=0;j<columnlength;j++){
                    this.matrix[k][j] = tmpRand.nextDouble();
                }
            }else{
                for(int j=0;j<columnlength;j++){
                    this.matrix[k][j] = 0.0;
                }
            }
        }
    }
    public Matrix(int rowlength,int columnlength){
        this(rowlength,columnlength,true);
    }

    public int getRowlength() {
        return rowlength;
    }

    public int getColumnlength() {
        return columnlength;
    }
    public double get(int rowindex,int columnindex){
        return this.matrix[rowindex][columnindex];
    }
    public void set(double value,int rowindex,int columnindex){
        this.matrix[rowindex][columnindex]=value;
    }
    @Override
    public String toString() {
        String str_size = "Size:["+this.rowlength+","+this.columnlength+"]\n";
        String str_mat = "";
        DecimalFormat df = new DecimalFormat("0.0000");
        for(int k=0;k<this.rowlength;k++){
            for(int j=0;j<this.columnlength;j++){
                str_mat += df.format(this.matrix[k][j]);
                str_mat += "\t";
            }
            str_mat+="\n";
        }
        return str_mat + str_size;
    }

    @Override
    public Matrix clone() throws CloneNotSupportedException {
        Matrix returnMat = new Matrix(this.rowlength,this.columnlength,false);
        for(int k =0;k<this.rowlength;k++){
            for(int j=0;j<this.columnlength;j++){
                returnMat.set(this.matrix[k][j],k,j);
            }
        }
        return returnMat;
    }
}
