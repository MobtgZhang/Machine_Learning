package com.mobtgzhang.utils.activates;

import com.mobtgzhang.utils.Activate;

public class SoftPlus extends Activate {
    public SoftPlus() {
        super("SoftPlus");
    }
    @Override
    public double innerFunc(double value) {
        return Math.log(1+Math.exp(value));
    }
}
