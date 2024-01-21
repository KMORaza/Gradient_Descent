package Stochastic_Gradient_Descent;
// Created: November 2022
public class SGDOptimizer {
    private double learningRate;
    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }
    public void updateParameters(LinearRegression model, double[] gradients) {
        model.updateWeights(gradients, learningRate);
    }
}
