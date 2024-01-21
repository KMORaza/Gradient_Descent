package Mini_Batch_Gradient_Descent;
// Created: November 2022
public class MBGDOptimizer {
    private double learningRate;
    public MBGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }
    public void updateParameters(LinearRegression model, Batch batch) {
        double[] predictions = new double[batch.getX().length];
        for (int i = 0; i < batch.getX().length; i++) {
            predictions[i] = model.predict(batch.getX()[i]);
        }
        double[] gradients = computeGradients(batch.getX(), batch.getY(), predictions);
        model.updateWeights(gradients, learningRate);
    }
    private double[] computeGradients(double[][] X, double[] y, double[] predictions) {
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