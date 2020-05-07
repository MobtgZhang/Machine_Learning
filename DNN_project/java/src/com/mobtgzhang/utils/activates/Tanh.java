package com.mobtgzhang.utils.activates;
import com.mobtgzhang.utils.Activate;
public class Tanh extends Activate {
    public Tanh() {
        super("Tanh");
    }
    @Override
    public double innerFunc(double value) {
        return Math.tan(value);
    }
}
