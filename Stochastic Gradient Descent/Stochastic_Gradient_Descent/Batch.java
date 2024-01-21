package Stochastic_Gradient_Descent;
// Created: November 2022
public class Batch {
    private double[][] X;
    private double[] y;
    public Batch(double[][] X, double[] y) {
        this.X = X;
        this.y = y;
    }
    public double[][] getX() {
        return X;
    }
    public double[] getY() {
        return y;
    }
}