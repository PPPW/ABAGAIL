package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.OptimizationAlgorithm;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.ConvergenceTrainer;

import java.util.*;
import java.io.*;
import java.text.*;


/**
 * Based on FourPeaksTest.java
 * @author Pei Wang
 */
public class myFourPeaks {
    /** The n value **/
    private static final int N = 50;
    /** The t value **/
    private static final int T = N / 10;

    //private static int trainingIterations = 1000;

    private static EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[4];
    private static String[] oaNames = {"RHC", "SA", "GA", "MIMIC"};

    private static DecimalFormat decFormat = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int i = Integer.parseInt(args[0]); // use the i'th algorithm
        int trainingIterations = Integer.parseInt(args[1]);

        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        oa[0] = new RandomizedHillClimbing(hcp);
        oa[1] = new SimulatedAnnealing(1E11, .95, hcp);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, gap);
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
         * Get iterations and training time vs problem size
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
               
        // check convergence
        double currentScore = -1;
        double lastScore = Double.MAX_VALUE;
        double threshold = 0.001;        
        int numConverge = trainingIterations / 10; 
        int currentConverge = 0;

        // the optimal score is 2*N - T - 1
        int currentNumOpt = 0;
        double optimalScore = 2*N - T -1;
        int numOpt = 1; 
        for(int i = 0; i < trainingIterations; i++) {    
            currentScore = oa.train();
            // if reaches the global optimal, stop
            if (currentScore == optimalScore) currentNumOpt++;
            //else currentNumOpt = 0;

            if (currentNumOpt == numOpt) { 
                //System.out.println("1-> "+ currentScore);
                return i;
            }
            // if converges, stop
            if (Math.abs(currentScore - lastScore) <= threshold) currentConverge++;
            else currentConverge = 0;

            if (currentConverge == numConverge) { 
                //System.out.println("2-> "+ currentScore);
                return i;
            }

            lastScore = currentScore;
            //System.out.println(oa.train());                        
        }
        //System.out.println("3-> "+ currentScore);
        return trainingIterations;        
    }
     
}
