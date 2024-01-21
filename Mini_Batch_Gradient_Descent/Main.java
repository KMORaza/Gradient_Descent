package Mini_Batch_Gradient_Descent;
// Created: November 2022
public class Main {
    public static void main(String[] args) {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
        double[] y = {3.0, 5.0, 7.0, 9.0, 11.0};
        Dataset dataset = new Dataset(X, y);
        LinearRegression model = new LinearRegression(X[0].length);
        MBGDOptimizer optimizer = new MBGDOptimizer(0.01);
        int epochs = 100;
        int batchSize = 2;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < dataset.getNumSamples() / batchSize; i++) {
                Batch batch = dataset.getBatch(batchSize);
                optimizer.updateParameters(model, batch);
            }
        }
        double[][] testX = {{6.0}, {7.0}};
        for (int i = 0; i < testX.length; i++) {
            double prediction = model.predict(testX[i]);
            System.out.println("Prediction for input " + testX[i][0] + ": " + prediction);
        }
    }
}