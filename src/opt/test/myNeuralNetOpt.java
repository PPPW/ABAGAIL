package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying ...
 * 
 * Based on Hannah Lau's AbaloneTest.java
 * @author Pei Wang
 */
public class myNeuralNetOpt {

    private static int inputLayer = 56, hiddenLayer = 0, outputLayer = 1;
    //private int trainingIterations = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int i = Integer.parseInt(args[0]); // use the i'th algorithm
        int trainingIterations = Integer.parseInt(args[1]);

        Instance[] instances = initializeInstances(30162, "src/opt/test/adult2train.csv");
        DataSet set = new DataSet(instances);

        Instance[] instancesTest = initializeInstances(15060, "src/opt/test/adult2test.csv");
        
        for(int j = 0; j < oa.length; j++) {
            networks[j] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[j] = new NeuralNetworkOptimizationProblem(set, networks[j], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E7, .75, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);
        
        // this for loop run one algorithm only

        //for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime; 
            double correct = 0, incorrect = 0;
            double correctTest = 0, incorrectTest = 0;
            train(oa[i], networks[i], oaNames[i], trainingIterations); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            // training set
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                actual = Double.parseDouble(instances[j].getLabel().toString());
                predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            
            // testing set
            for(int j = 0; j < instancesTest.length; j++) {
                networks[i].setInputValues(instancesTest[j].getData());
                networks[i].run();

                actual = Double.parseDouble(instancesTest[j].getLabel().toString());
                predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correctTest++ : incorrectTest++;
            }

            results +=  "\nResults for " + oaNames[i] 
                + "%\nTraining time: " + df.format(trainingTime) + " seconds\n"
                + "Training correct: " + correct + " instances.\n" 
                + "Training incorrect: " + incorrect + " instances.\n"
                + "Training accuracy: " 
                + df.format(correct/(correct+incorrect)*100) 
                + "\nTesting correct: " + correctTest + " instances." 
                + "\nTesting incorrect: " + incorrectTest + " instances.\n"
                + "Testing accuracy: " 
                + df.format(correctTest/(correctTest+incorrectTest)*100);
                
        //}
        
        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations) {
        System.out.println(trainingIterations);
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            double error = 1 / oa.train();
            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances(int numOfRows, String fileName) {
        double[][][] attributes = new double[numOfRows][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[inputLayer]; 
                attributes[i][1] = new double[1]; // 1 label

                for(int j = 0; j < inputLayer; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                // last one in a row is the label
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // convert {1, 2} into {0, 1}
            instances[i].setLabel(new Instance(attributes[i][1][0] == 1 ? 0 : 1));
        }

        return instances;
    }
}
