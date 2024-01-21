package Mini_Batch_Gradient_Descent;
// Created: November 2022
public class LinearRegression {
    private double[] weights;
    private double bias;
    public LinearRegression(int inputSize) {
        this.weights = new double[inputSize];
        this.bias = Math.random();
    }
    public double predict(double[] X) {
        double result = bias;
        for (int i = 0; i < weights.length; i++) {
            result += weights[i] * X[i];
        }
        return result;
    }
    public void updateWeights(double[] gradients, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * gradients[i];
        }
        bias -= learningRate * gradients[weights.length];
    }
}