package com.mobtgzhang.utils.activates;

import com.mobtgzhang.utils.Activate;

public class Sigmoid extends Activate {
    public Sigmoid() {
        super("Sigmoid");
    }
    @Override
    public double innerFunc(double value) {
        return 1.0/(1+Math.exp(value));
    }
}
