package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.OptimizationAlgorithm;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * @author Pei Wang
 */
public class myTravelingSalesman {
    /** The n value */
    private static final int N = 20;
   
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[4];
    private static String[] oaNames = {"RHC", "SA", "GA", "MIMIC"};

    private static DecimalFormat decFormat = new DecimalFormat("0.000");

    
    public static void main(String[] args) {
        int i = Integer.parseInt(args[0]); // use the i'th algorithm
        int trainingIterations = Integer.parseInt(args[1]);
        
        Random random = new Random(0);
        // create the random points
        double[][] points = new double[N][2];
        for (int j = 0; j < points.length; j++) {
            points[j][0] = random.nextDouble();
            points[j][1] = random.nextDouble();
            System.out.println(points[j][0] + " " + points[j][1]);
        }
        System.exit(0);
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);        
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        oa[0] = new RandomizedHillClimbing(hcp);
        oa[1] = new SimulatedAnnealing(1E11, .95, hcp);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, gap);

        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        oa[3] = new MIMIC(200, 20, pop);

        /*************************************************************
         * check how many iterations it takes for convergence.
         * 
         **************************************************************/
        /*
        int numToRun = 20;
        for (int j = 0; j < numToRun; j++) {
            oa[0] = new RandomizedHillClimbing(hcp);
            oa[1] = new SimulatedAnnealing(1E11, .95, hcp);
            oa[2] = new StandardGeneticAlgorithm(200, 100, 10, gap);
            oa[3] = new MIMIC(200, 20, pop);

            double start = System.nanoTime(), end, trainingTime;
            System.out.print(train(oa[i], oaNames[i], trainingIterations) + " " +
                             ef.value(oa[i].getOptimal()) + " "
                             );
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            System.out.println(decFormat.format(trainingTime));
        }
        */
        /*************************************************************
         * Get scores vs iterations, and training time.
         * 
         **************************************************************/
             
        System.out.println("Results for: " + oaNames[i]);
        double start = System.nanoTime(), end, trainingTime;
        System.out.println("Problem size: " + N);
        System.out.println("Iterations: " + train(oa[i], oaNames[i], trainingIterations)); 
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        System.out.println("Training time: " + decFormat.format(trainingTime));
        System.out.println("Final score:  " + ef.value(oa[i].getOptimal()));
        
        System.out.println(oa[i].getOptimal());        
        
        /*
        // the ConvergenceTrainer has a problem: if it stuck at local minimum for 
        // even one iteration, it's recoginized as converged.
        ConvergenceTrainer ct = new ConvergenceTrainer(oa[i], 0.01, 100000);
        ct.train();
        System.out.println(ct.getIterations() + " " + ef.value(oa[i].getOptimal()));
        */
    }

    private static int train(OptimizationAlgorithm oa, String oaName, int trainingIterations) {
        //System.out.println("Max iterations: ", trainingIterations);
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
               
        double currentScore;
        double lastScore = Double.MAX_VALUE;
        double threshold = 0.001;
        int currentNumOpt = 0;
        int numOpt = trainingIterations / 10; 
        for(int i = 0; i < trainingIterations; i++) {    
            currentScore = oa.train();
            if (Math.abs(currentScore - lastScore) <= threshold) currentNumOpt++;
            else currentNumOpt = 0;

            if (currentNumOpt == numOpt) {                
                return i;
            }

            lastScore = currentScore;
            //System.out.println(oa.train());                        
        }
        return trainingIterations; 
    }
}
