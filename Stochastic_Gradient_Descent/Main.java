package Stochastic_Gradient_Descent;
// Created: November 2022
public class Main {
    public static void main(String[] args) {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
        double[] y = {3.0, 5.0, 7.0, 9.0, 11.0};
        Dataset dataset = new Dataset(X, y);
        LinearRegression model = new LinearRegression(X[0].length);
        SGDOptimizer optimizer = new SGDOptimizer(0.01);
        int epochs = 100;
        int batchSize = 2;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < dataset.getNumSamples() / batchSize; i++) {
                Batch batch = dataset.getBatch(batchSize);
                double[] predictions = new double[batchSize];
                for (int j = 0; j < batchSize; j++) {
                    predictions[j] = model.predict(batch.getX()[j]);
                }
                double[] gradients = computeGradients(batch.getX(), batch.getY(), predictions);
                optimizer.updateParameters(model, gradients);
            }
        }
        double[][] testX = {{6.0}, {7.0}};
        for (int i = 0; i < testX.length; i++) {
            double prediction = model.predict(testX[i]);
            System.out.println("Prediction for input " + testX[i][0] + ": " + prediction);
        }
    }
    private static double[] computeGradients(double[][] X, double[] y, double[] predictions) {
        int numFeatures = X[0].length;
        double[] gradients = new double[numFeatures + 1];
        for (int i = 0; i < X.length; i++) {
            double error = predictions[i] - y[i];
            gradients[numFeatures] += error;
            for (int j = 0; j < numFeatures; j++) {
                gradients[j] += error * X[i][j];
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            gradients[j] /= X.length;
        }
        gradients[numFeatures] /= X.length;
        return gradients;
    }
}