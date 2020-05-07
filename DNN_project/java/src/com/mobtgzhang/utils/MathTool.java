package com.mobtgzhang.utils;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;

public class MathTool {
    public static Matrix add(Matrix matA,Matrix matB) throws Exception {
        if(matA.getRowlength()!=matB.getRowlength()&&matA.getColumnlength()!=matB.getColumnlength()){
            String message = "The size:"+"("+matA.getRowlength()+","+matA.getColumnlength()+")"
                    + "and ("+matB.getRowlength()+","+matB.getColumnlength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)+matB.get(k,j),k,j);
            }
        }
        return returnMat;
    }
    public static Matrix sub(Matrix matA,Matrix matB) throws Exception {
        if(matA.getRowlength()!=matB.getRowlength()&&matA.getColumnlength()!=matB.getColumnlength()){
            String message = "The size:"+"("+matA.getRowlength()+","+matA.getColumnlength()+")"
                    + "and ("+matB.getRowlength()+","+matB.getColumnlength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)-matB.get(k,j),k,j);
            }
        }
        return returnMat;
    }
    public static Matrix mul(Matrix matA,Matrix matB) throws Exception {
        if(matA.getRowlength()!=matB.getRowlength()&&matA.getColumnlength()!=matB.getColumnlength()){
            String message = "The size:"+"("+matA.getRowlength()+","+matA.getColumnlength()+")"
                    + "and ("+matB.getRowlength()+","+matB.getColumnlength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)*matB.get(k,j),k,j);
            }
        }
        return returnMat;
    }
    public static Matrix div(Matrix matA,Matrix matB) throws Exception {
        if(matA.getRowlength()!=matB.getRowlength()&&matA.getColumnlength()!=matB.getColumnlength()){
            String message = "The size:"+"("+matA.getRowlength()+","+matA.getColumnlength()+")"
                    + "and ("+matB.getRowlength()+","+matB.getColumnlength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)/matB.get(k,j),k,j);
            }
        }
        return returnMat;
    }
    public static Matrix add(Matrix matA,double value)  {
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)+value,k,j);
            }
        }
        return returnMat;
    }
    public static Matrix sub(Matrix matA,double value) {
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)-value,k,j);
            }
        }
        return returnMat;
    }
    public static Matrix mul(Matrix matA,double value) {
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)*value,k,j);
            }
        }
        return returnMat;
    }
    public static Matrix div(Matrix matA,double value) {
        int rowlength = matA.getRowlength();
        int columnlength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(matA.get(k,j)/value,k,j);
            }
        }
        return returnMat;
    }
    public static Vector add(Vector matA, Vector matB) throws Exception {
        if(matA.getLength()!=matB.getLength()){
            String message = "The size:"+"("+matA.getLength()+")"
                    + "and ("+matB.getLength()+") don't match!";
            throw new Exception(message);
        }
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)+matB.get(k),k);
        }
        return returnMat;
    }
    public static Vector sub(Vector matA, Vector matB) throws Exception {
        if(matA.getLength()!=matB.getLength()){
            String message = "The size:"+"("+matA.getLength()+")"
                    + "and ("+matB.getLength()+") don't match!";
            throw new Exception(message);
        }
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)-matB.get(k),k);
        }
        return returnMat;
    }
    public static Vector mul(Vector matA, Vector matB) throws Exception {
        if(matA.getLength()!=matB.getLength()){
            String message = "The size:"+"("+matA.getLength()+")"
                    + "and ("+matB.getLength()+") don't match!";
            throw new Exception(message);
        }
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)*matB.get(k),k);
        }
        return returnMat;
    }
    public static Vector div(Vector matA, Vector matB) throws Exception {
        if(matA.getLength()!=matB.getLength()){
            String message = "The size:"+"("+matA.getLength()+")"
                    + "and ("+matB.getLength()+") don't match!";
            throw new Exception(message);
        }
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)/matB.get(k),k);
        }
        return returnMat;
    }
    public static Vector add(Vector matA,double value){
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)+value,k);
        }
        return returnMat;
    }
    public static Vector sub(Vector matA,double value){
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)-value,k);
        }
        return returnMat;
    }
    public static Vector mul(Vector matA,double value){
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)*value,k);
        }
        return returnMat;
    }
    public static Vector div(Vector matA,double value) {
        int length = matA.getLength();
        Vector returnMat = new Vector(length,false);
        for(int k=0;k<length;k++){
            returnMat.set(matA.get(k)/value,k);
        }
        return returnMat;
    }
    public static Matrix add(Matrix mat,Vector vec) throws Exception {
        if(mat.getColumnlength()!=vec.getLength()){
            String message = "The size:"+"("+mat.getRowlength()+","+mat.getColumnlength()+")"
                    + "and ("+vec.getLength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = mat.getRowlength();
        int columnlength = mat.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(mat.get(k,j)+vec.get(j),k,j);
            }
        }
        return returnMat;
    }
    public static Matrix sub(Matrix mat,Vector vec) throws Exception {
        if(mat.getColumnlength()!=vec.getLength()){
            String message = "The size:"+"("+mat.getRowlength()+","+mat.getColumnlength()+")"
                    + "and ("+vec.getLength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = mat.getRowlength();
        int columnlength = mat.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                returnMat.set(mat.get(k,j)-vec.get(j),k,j);
            }
        }
        return returnMat;
    }
    public static Matrix dot(Matrix matA,Matrix matB) throws Exception {
        if(matA.getColumnlength()!=matB.getRowlength()){
            String message = "The size:"+"("+matA.getRowlength()+","+matA.getColumnlength()+")"
                    + "and ("+matB.getRowlength()+","+matB.getColumnlength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = matA.getRowlength();
        int columnlength = matB.getColumnlength();
        int tmplength = matA.getColumnlength();
        Matrix returnMat = new Matrix(rowlength,columnlength,false);
        for(int k=0;k<rowlength;k++){
            for(int j=0;j<columnlength;j++){
                double value = 0;
                for(int i=0;i<tmplength;i++){
                    value += matA.get(k,i)*matB.get(i,j);
                }
                returnMat.set(value,k,j);
            }
        }
        return returnMat;
    }
    public static Vector dot(Vector vec,Matrix mat) throws Exception {
        if(vec.getLength()!=mat.getRowlength()){
            String message = "The size:"+"("+vec.getLength()+")"
                    + "and ("+mat.getRowlength()+","+mat.getColumnlength()+") don't match!";
            throw new Exception(message);
        }
        int rowlength = mat.getRowlength();
        int columnlength = mat.getColumnlength();
        Vector returnVec = new Vector(columnlength);

        for(int j=0;j<columnlength;j++){
            double value = 0;
            for(int k=0;k<rowlength;k++){
               value += vec.get(k)*mat.get(k,j);
            }
            returnVec.set(value,j);
        }
        return returnVec;
    }
}
