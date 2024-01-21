package Stochastic_Gradient_Descent;
// Created: November 2022
import java.util.Random;
import java.util.Random;
public class Dataset {
    private double[][] X;
    private double[] y;
    private int numSamples;
    public Dataset(double[][] X, double[] y) {
        this.X = X;
        this.y = y;
        this.numSamples = X.length;
    }
    public int getNumSamples() {
        return numSamples;
    }
    public Batch getBatch(int batchSize) {
        Random rand = new Random();
        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indices[i] = rand.nextInt(numSamples);
        }
        return new Batch(getSubset(X, indices), getSubset(y, indices));
    }
    private double[][] getSubset(double[][] array, int[] indices) {
        double[][] subset = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            subset[i] = array[indices[i]];
        }
        return subset;
    }
    private double[] getSubset(double[] array, int[] indices) {
        double[] subset = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            subset[i] = array[indices[i]];
        }
        return subset;
    }
}