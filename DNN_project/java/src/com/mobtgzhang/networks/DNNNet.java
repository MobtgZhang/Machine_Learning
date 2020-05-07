package com.mobtgzhang.networks;

import com.mobtgzhang.matrix.Matrix;
import com.mobtgzhang.matrix.Vector;
import com.mobtgzhang.utils.Activate;
import com.mobtgzhang.utils.MathTool;

public abstract class DNNNet {
    protected int[] dim_list;
    protected int num_layers;
    protected String act_name;
    protected Activate actFunc;
    protected Linear [] linears;
    public DNNNet(int [] dim_list,String act_name) throws Exception {
        this.dim_list = dim_list;
        this.num_layers = dim_list.length-1;
        this.act_name = act_name;
        this.actFunc = MathTool.selectAct(act_name);
        //create layers
        this.linears = new Linear[dim_list.length-1];
        for(int k=0;k<dim_list.length-1;k++){
            this.linears[k] = new Linear(dim_list[k],dim_list[k+1]);
        }
    }
    protected Matrix forward(Matrix input) throws Exception {
        Matrix output = input;
        for(int k=0;k<this.num_layers-1;k++){
            Matrix hidden = this.linears[k].forward(output);
            output = this.actFunc.forward(hidden);
        }
        return this.linears[this.num_layers-1].forward(output);
    }
    protected Vector forward(Vector input) throws Exception {
        Vector output = input;
        for(int k=0;k<this.num_layers-1;k++){
            Vector hidden = this.linears[k].forward(output);
            output = this.actFunc.forward(hidden);
        }
        return this.linears[this.num_layers-1].forward(output);
    }
    public abstract void backward();
}
