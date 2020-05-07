package com.mobtgzhang.utils.activates;

import com.mobtgzhang.utils.Activate;

public class Relu extends Activate {
    private double alpha;
    public Relu(double alpha) {
        super("Relu");
        this.alpha = alpha;
    }
    public Relu() {
        this(1.0);
    }
    @Override
    public double innerFunc(double value) {
        if (value<0){
            return 0;
        }else {
            return value*this.alpha;
        }
    }
}
