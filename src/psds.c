/*-------------------------------------------------------------------------------------------------------------------*/
/*                                                                                                                   */
/*  "Study hard what interests you the most in the most undisciplined, irreverent and original manner possible."     */
/*     -- Richard Feynman                                                                                            */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/*  DEFAULT HYPERPARAMETER VALUES                                                                                    */
/*                                                                                                                   */
#define  INPUT_LAYER_CNT     (28*28)+1    /* +1 is for bias unit, the highest numbered element                       */
#define  HIDDEN0_LAYER_CNT   204+1        /* +1 is for bias unit (highest numbered element)                          */
                                          /* Best values (?): 205, 184, 177, 170, 84                                 */
#define  OUTPUT_LAYER_CNT    10           /* since nothing follows output layer, no extra bias unit                  */
                                          /*                                                                         */
#define  RELOAD_WEIGHTS      0            /*  1: use weights previously saved to loadWeights.c (requires recompile)  */
#define  MAX_HP_SWEEPS       1            /*  Max number of hyperparameter sweeps. 1: normal run (no sweep)          */
#define  MAX_EPOCHS          6            /*  1 ...                                                                  */
#define  LEARNING_RATE       0.19000000   /*  (eta) -- initial learning rate. Reasonable value: 0.150                */
#define  MOMENTUM            0.10         /*  (alpha) -- momentum coefficient                                        */
#define  REGULARIZE_TYPE     REG_L2       /*  What kind, if any, regularization to use to calculate output error     */
#define  K_REGULARIZE        0.000000     /*  (lambda) -- L2 norm regularizaton coefficient                          */
#define  STOP_VALID_SCORE    1.9340000    /*  Stop of validation score reaches this value (and early stop enabled)   */
#define  TRAIN_SUBSET_SIZE   (42000/4)*4  /*  No. images to be used in TRAINING the net   [29400 = 42K * 0.7]        */
#define  VALID_SUBSET_SIZE   (42000/4)*4  /*  No. images to be used in VALIDATING the net [12600 = 42K * 0.3]        */
#define  SKIP_SUBSET_SIZE    (42000/4)*0  /*  Number of images to skip, to implement k-fold validation; 0=disable    */
#define  UPDATE_RULE_TYPE    UR_ONLINE    /*  see psds.h for a list of defined values                                */
#define  MINIBATCH_SIZE      100          /*  For minibatch updates: number of training cases per weight update      */
#define  NOISE               0.00         /*  Amount of noise to add to image for 2nd pass; 0.0 = disable            */
#define  WEIGHT_DECAY        0.00000      /*  Weight decay per epoch; 0.00 = no weight decay. Try 0.00001 or lower   */
#define  RANDOM_WT_SCALE     0.1960       /*  Used for scaling random numbers used to initialize weights             */
/*  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - */
#define  DEBUG_LEVEL         1            /*  0: no debug; 1: basic; 2: verbose                                      */
#define  PROCESS_PRIO        19           /*  Priority of this processs; -20=hoggiest ... 19=nicest :)               */
                                          /*  Anything hoggier than 8 needs superuser privs (run with sudo)          */
#define  FB_IMP              FBI_JB       /*  FBI_DEFAULT, FBI_SAS, FBI_JB, FBI_MM, ...                              */
#define  BACKPROP_DEV_MODE   0            /*  1: force small dataset for development (essentially a unit test)       */
#define  RUN_DWN_GRAD_LMT    1            /*                                                                         */
#define  WEIGHT_TWEAK        0.00001      /*  0.0001 is a reasonable value                                           */
                                          /*                                                                         */
/* FILE PATHS                                                                                                        */
#define  TRAIN_FILE_PATH     "../dataset/train.csv"       /*  path to training data CSV file                         */
#define  TEST_FILE_PATH      "../dataset/test.csv"        /*  path to test data CSV file                             */
#define  TRAIN_GRAPH_PATH    "../graphs/training.html"    /*                                                         */
#define  VALID_GRAPH_PATH    "../graphs/validation.html"  /*                                                         */
#define  HPSWEEP_GRAPH_PATH  "../graphs/hpSweep.html"     /*                                                         */
                                                          /*                                                         */
/*-------------------------------------------------------------------------------------------------------------------*/
/* REVISION HISTORY                                                                                                  */
/*                                                                                                                   */
/* 05 APR 2017  SAS. Achieved my new goal of >= 96%: got 96.914% today. 6 epochs rather than 4 as in most previous   */
/*              submissions; fine tuned random weight initialization. THE BEST PART is, I finally beat the two       */
/*              benchmarks: "Random forest" at 96.829%, and "KNN, K=10" at 96.557%.  :-)                             */
/*              I had not previously submitted anything past epoch 4 because I had more of an issue with overfitting */
/*              before. Eithe rthis belief was a mistake, or something else changed.                                 */
/*                                                                                                                   */
/* 04 APR 2017  SAS. Achieved my goal of >= 95% Kaggle score (95.714%), but now I want to see if I can go higher...  */
/*                                                                                                                   */
/* 01 APR 2017  SAS. FINALLY got backprop implemented and working. After 3 epochs lasting just 3m 51s, made a Kaggle */
/*              digit recognition submission which scored 0.91329. The previous score of 0.88729 took almost a full  */
/*              day to run just a single epoch (using tweek method). Next: a bit of cleanup, then hit the TODOs.     */
/*                                                                                                                   */
/* 26 MAR 2017  SAS. Created simple HTML5 canvas-based graphing utility (graph.c/graph.h) admn plumbed training      */
/*              error to it so I can watch error graphically in real time. Added simple early stopping if the error  */
/*              value goes up at all -- crude, but a starting place. Also, just for the record, I still have NO      */
/*              hidden layer backprop implemented, only output layer, so I am still just laming.                     */
/*                                                                                                                   */
/* 23 MAR 2017  SAS. Spent the past several days on a deep dive into Rumelhart/Hinton/Willaims backprop algorithm.   */
/*              Having previously become utterly stuck, I am now gutting and rewriting the hidden layer portion      */
/*              of backward(), which is the backpropagation routine. I now have as reference a Google Docs           */
/*              spreadsheet which makes all backprop calculations for a simple 2x2x2 neural net, inspired by         */
/*              the excellent tutorial by Matt Mazur.                                                                */
/*                                                                                                                   */
/* 14 MAR 2017  SAS. Initial creation. Frustrated by the scarcity and low quality of existing machine learning       */
/*              libraries suitable for embedding on small hardware, specifically written entirely in C.              */
/*              Inspirational (but insufficiant) examples are libdeep, FANN, and rimstar.                            */
/*              INITIALLY, this entire project will be hard-coded to solve the Kaggle Digit Recognition challenge.   */
/*              When (and if) successful, we will generalize it into the beginning of a truely general purpose       */
/*              library. Wish me luck.                                                                               */
/*              Initially, we will use a single hidden layer.                                                        */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/* SUBMISSIONS -- "PS1", Kaggle ID 535192                                                                            */
/*                                                                                                                   */
/* 007 / 0.61243  Mon Mar 27. This may LOOK like a bad result, but it was a sanity check after MASSIVE work on my    */
/*                code. Consider the following numbers: training set of ONLY 2940 images; validation set 1260        */
/*                images. ONLY 1 pass (epoch) through the test set. CPU time ONLY 00d:00h:36m:29s !!!                */
/*                Validation set score: 0.60372747 (interestingly, my Kaggle score on the test data is a bit better  */
/*                than my own score on my chosen validation set). This is the first time I have even HAD a           */
/*                validation score to compare with Kaggle! By the way, the pruning code I added to forward() this    */
/*                evening resulted in a 23x speedup! So instead of 17 seconds, images are now processed in < 3/4     */
/*                second each.                                                                                       */
/*                                                                                                                   */
/* 006 / 0.09014  Sun Mar 26. Code badly broken; no seperate validation set used yet; what can I say.                */
/*                                                                                                                   */
/* 005 / 0.11100  Sun Mar 26. Just a sanity check; didn't expect anything different as no progress made on backprop  */
/*                implementation (although it was fun to watch the training error curve on the new graph). However   */
/*                something is badby broken -- looking at X.html, I see that all labels are (-1).                    */
/*                                                                                                                   */
/* 004 / 0.80329  Sun Mar 19. In weight initialization, add scaling by sqrt of fan-in, ala Hinton and others. As a   */
/*                result, the total error now decreases monotonically until epoch 120 or so, whereas previously it   */
/*                started small, shot up, then wandered down again. MAX_EPOCHS = 120, all other HP's unchanged.      */
/*                Total error @ epoch 120: 0.03542922  CPU time: 00d:00h:20m:57s                                     */
/*                                                                                                                   */
/* 003 / 0.79686  Sun Mar 19. Hyperparameter changes only, then went to bed, then submitted with coffee.             */
/*                MAX_EPOCHS = 1000, LEARNING_RATE = 0.02. CPU time: 00d:00h:53m:31s                                 */
/*                                                                                                                   */
/* 002 / 0.79100  Sun Mar 19. Same basic setup as before, but I forgot to remove the first line (labels) from the    */
/*                test data, same as I already do for the training data, so everything in the submission file was    */
/*                presumable shifted by 1, and I previously had to manually remove the 28001st entry. Duh.           */
/*                CPU time: 00d:00h:08m:46s                                                                          */
/*                                                                                                                   */
/* 001 / 0.10000  Sun Mar 19. MAX_EPOCHS = 50, 1 hidden layer with 100 units, LEARNING_RATE = 0.10                   */
/*                I know backprop is broken (incorrent hidden layer weigh update; bias                               */
/*                weights are not updated AT ALL; etc. But I wanted to make a first submission now that I have the   */
/*                code working end-to-end, and before I go to bed (ats after 4:00 AM). Did no manual check of the    */
/*                submission file apart from eyeballing to make sure the indexes were current and all 28K entries    */
/*                were there, and there was some reasonable distribution of 0...9. CPU time: 00d:00h:08m:47s         */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/* HYPERPARAMETER ZOO -- Promising and interesting hyperparameters                                                   */
/*                                                                                                                   */
/* Any configuration with 100 hidden units seems to top out at about 80% validation score, give or take.             */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/* (3rd son) Tzu                                                                                                     */
/* "I am not going to fight that battle, but if YOU want to, go right ahead -- god forbid I contradict you!"         */
/*   -- ppb, 29 March 2017, 9:05 pm, after I chastised spud for coming back downstars 20 minutes past his            */
/*      bedtime, and she had casually engaged with and chatted with him.                                             */
/*-------------------------------------------------------------------------------------------------------------------*/
/* TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO   TODO     */
/*                                                                                                                   */
/*  [ ] FIX L2 regularization of weight values                                                                       */
/*                                                                                                                   */
/*  [ ] Perform a hyperparam sweep of random weight init magnitudes.                                                 */
/*  [ ] Perform a hyperparam sweep of srand() seed values                                                            */
/*  [ ] Training error graph seems to be going the wrong way (should be converging to zero) -- investigate           */
/*  [ ] Fix status information display, during training and test. Broken/messy/incomplete.                           */
/*  [ ] Update submission html file generation                                                                       */
/*  [ ] Perform momentum sweep                                                                                       */
/*  [ ] Look into using CUDAMat or other CUDA-enabled maxrix library to speed things up when available               */
/*  [ ] For the purposes of graphing, validation and training error must be calculated with the same units; I also   */
/*      need a 3rd metric, which is accuracy. Generate that 3rd graph (or combine all 3 into one graph?), or...?     */
/*       ==> https://en.wikipedia.org/wiki/Cross-validation_(statistics)                                             */
/*  [ ] Implement mini batches [ref 3]                                                                               */
/*  [ ] In the graph html, place text well below the canvas which details the hyperparameters, epoch, ...            */
/*  [ ] Add some forgiveness to early stopping, such that it tries modifying tweek values etc. before giving up      */
/*  [ ] Individually name weight dump files with epoch number and validation score                                   */
/*  [ ] Add the ability to store and reload error values so that after restart, psds can continue to generate        */
/*      graphs containing a full set of error values.                                                                */
/*  [ ] Add ability to load key hyperparameters from a config file                                                   */
/*  [ ] Add a csv file method of storing and reloading weights, to prevent having to recompile each time             */
/*  [ ] Investigate Linux signals and signal handling, create simple way to send commands to the process             */
/*  [ ] Finish BACKPROP_DEV_MODE (2x2x2 unit testing mode) and verify all backprop math                              */
/*  [ ] Add main menu providing the following options: Set hyperparameters; select training file; train;             */
/*      weights to file; load weights from file; test & generate summary file                                        */
/*  [ ] Dump test images with most probable labels, like how I presently do with training images & labels. This      */
/*      could include confidence values for each classification                                                      */
/*  [ ] Implement dropout (or DropConnect, or...)     https://en.wikipedia.org/wiki/Dropout_(neural_networks)        */
/*  [ ] Consider switching to ReLU rather than sigmoid activation                                                    */
/*  [X] Perform sweep of # of hidden0 units                                                                          */
/*  [X] Perform learning rate sweep                                                                                  */
/*  [X] Restructure and modularize the main loop, separating (at a minimum) validation and test                      */
/*      into functions so as to allow their use in various combinations. This will facilitate minibatches,           */
/*      selective generation of weight files, mini batches, etc.                                                     */
/*  [X] Implement k-fold cross-validation                                                                            */
/*  [X] Make the number of hidden layers dynamically configurable. This will allow the automated                     */
/*      search for optimum hyperparameters.                                                                          */
/*  [X] Consider loading all image data into memory at the start (only 28*28*42000 bytes = 32 Mb). This will         */
/*      somewhat improve performance, but more importantly will make things like k-fold cross-validation simple.     */
/*      This also allows learning from the training set in random order, which can help apparently                   */
/*  [X] Automate hyperparameter search: number of hidden units, tweak value, ...                                     */
/*  [X] Implement bias weight update in backprop() (not just backprop_sas()                                          */
/*  [X] Generate graphs of BOTH validation score and net output error.                                               */
/*  [X] Date & timestamp submission file. At the same time, create another date/timestamped metaparameter dump file  */
/*  [X] Consider generating artificial data from the training samples to increase training data                      */
/*      --> but only after exhausting ALL available original training images)                                        */
/*      --> see sandbox in directory psds/synth                                                                      */
/*  [X] Consider adding more hidden layers                                                                           */
/*  [X] Try different number of hidden units (initially 100) in hidden layer 0.                                      */
/*  [X] Graph X axis should be epoch, not seconds                                                                    */
/*  [X] Implement forward_jb() and backward_jb() from John Bullinaria's nn code (ACTIVE/nn)                          */
/*  [X] Add ability to set niceness at start (19 is lowest priority; default is 12)                                  */
/*  [X] Add confidence metric calculation to getDigitClassification(), add confidence property to net_t              */
/*       ==> see [ref 8]                                                                                             */
/*  [X] Try changing "run 100 steps down the gradient" to 500, 10, 0                                                 */
/*  [X] Normalize pixel values!! (0 ... 255  --->  0.0 ... 1.0 )                                                     */
/*  [X] Re-implement bias weights as just another neuron, which is just stuck at 1. [ref 3] states: "Units can be    */
/*      given biases by introducing an extra input to each unit which always has a value of 1."                      */
/*  [X] Bias weights are not being stored/reloaded! DUH! ==> Fixed as side effect of integrating bias units          */
/*  [X] Display all hyperparameters at startup                                                                       */
/*  [X] Break training data into training and validation sets, and keep track of validation score. This is the one   */
/*      that really matters.                                                                                         */
/*  [X] Implement early stopping to prevent overfitting                                                              */
/*  [X] Smarter calculation of random initial weights -- see Hinton's lecture where he suggests proportionality      */
/*      to fan-in, or some such                                                                                      */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/* ARTIFICIAL DATA                                                                                                   */
/*                                                                                                                   */
/*   1. Add random noise to the backgrounds of digits                                                                */
/*   2. Scaling and rotation                                                                                         */
/*   3. Bluring                                                                                                      */
/*   4. Warping, elastic distorsions - shifting image around in wave patterns -- superimpose grid squares, then      */
/*      perturb with horizontal and vertical grid bars in waves etc.                                                 */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/* MISCELLANEOUS                                                                                                     */
/*                                                                                                                   */
/* This wikipedia article [https://en.wikipedia.org/wiki/MNIST_database] describes using 2 layer neural nets to      */
/* classify handwritten digits in the MNIST database. They were both configured 784-800-10 (800 hidden units!)       */
/* deep reinforcement learning?                                                                                      */
/* Rectified Linear Units (ReLU)?                                                                                    */
/* "Learning multiple layers of features" -- Geoff Hinton, nn4ml 3d                                                  */
/*                                                                                                                   */
/* First-order methods such as SGD, Adam, RMSprop, Adadelta, or Adagrad: backpropagation in first-order methods      */
/* requires first-order derivative.                                                                                  */
/*   http://datascience.stackexchange.com/questions/12744/backpropagation-in-second-order-methods-would-relu-derivative-be-0-and-what-i  */
/* "An overview of gradient descent optimization algorithms", Sebastian Ruder (paper)                                */
/* hinton-nn4ml-lecture-6a--overview-of-mini-batch-gradient-descent.pdf                                              */
/*                                                                                                                   */
/* The Second Machine Age -- audiobook                                                                               */
/* [ 2.0 TB Volume/public/multimedia/Erik Bryn... ]                                                                  */
/*                                                                                                                   */
/* train.csv statistics                                                                                              */
/*   first  100 training cases: [ std dev:  3.30 ]                                                                   */
/*     0:9     1:16   2:11    3:11   4:11    5:6     6:9    7:7     8:6    9:14                                      */
/*   first 1000 training cases: [ std dev: 10.32 ]                                                                   */
/*     0:107   1:96   2:124   3:90   4:102   5:89    6:97   7:105   8:93   9:97                                      */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/
/* REFERENCES                                                                                                        */
/*                                                                                                                   */
/*   1. Stephen C. Welch: "Neural Networks Demystified" 7-part YouTube video tutorial, 2014                          */
/*   2. Matt Mazur: "A Step by Step Backpropagation Example", online tutorial, 2015                                  */
/*   3. Rummelhart, D. E., Hinton, G. E., & Williams, R. J.: "Learning representations by back-propagating errors",  */
/*      1986                                                                                                         */
/*   4. Hinton, G. E., Osindero, O., & Teh, Y.: "A fast learning algorithm for deep belief nets", 2006               */
/*   5. John Bullinaria: "Step by Step Guide to Implementing a Neural Network in C",                                 */
/*      http://www.cs.bham.ac.uk/~jxb/INC/nn.html                                                                    */
/*   6. "Chapter 6: Deep learning", http://neuralnetworksanddeeplearning.com/chap6.html                              */
/*   7. Zou, W., Li, Y., & Tang, A.: "Effects of the Number of Hidden Nodes Used in a Structured-based Neural        */
/*      Network on the Reliability of Image Classification"                                                          */
/*   8. Zaragoza, H., & d'Alache-Buc., F.: "Confidence Measures for Neural Network Classifiers"                      */
/*   9. Anderson, T., & Martinez, T.: "Cross Validation and MLP Architecture Selection"                              */
/*  10. Geman, S., Bienenstock, E., & Doursat, R.: "Neural Networks and the Bias/Variance Dilemma", 1992             */
/*  11. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.: "Dropout: A Simple Way to    */
/*      Prevent Neural Networks from Overfitting", 2014                                                              */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/

/*
       int pthread_setaffinity_np(pthread_t thread, size_t cpusetsize,
                                  const cpu_set_t *cpuset);
       int pthread_getaffinity_np(pthread_t thread, size_t cpusetsize,
                                  cpu_set_t *cpuset);
*/

#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <sys/time.h>
#include <sys/resource.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "psds.h"
#include "graph.h"

#define  TRAIN_FILE_SIZE         42000
#define  TEST_FILE_SIZE          28000
#define  MAX_LINE_SIZE           10000  /* training file line buffer size */
#define  MAX_INPUT_LAYER_CNT     (28*28)+1
#define  MAX_HIDDEN0_LAYER_CNT   1000
#define  MAX_OUTPUT_LAYER_CNT    10

#if( BACKPROP_DEV_MODE )
  /* For use with  https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/  */
  #undef   INPUT_LAYER_CNT
  #define  INPUT_LAYER_CNT           2+1        /* +1 is for bias unit (highest numbered element) */
  #undef   HIDDEN0_LAYER_CNT
  #define  HIDDEN0_LAYER_CNT         2+1        /* +1 is for bias unit (highest numbered element) */
  #undef   OUTPUT_LAYER_CNT
  #define  OUTPUT_LAYER_CNT          2          /* since nothing follows output layer, no extra bias unit */
  /* Redefine some hyperparameters for the test */
  #undef   LEARNING_RATE
  #define  LEARNING_RATE             0.5
  #undef   MAX_EPOCHS
  #define  MAX_EPOCHS                1
#endif


typedef struct _hyperparam_t
{
  int        inputLayerCnt;               /* Number of inputs; add 1 for bias unit, which is highest index elem */
  int        hidden0LayerCnt;             /* Number of hidden units; add 1 for bias unit... */
  int        outputLayerCnt;              /* since nothing follows output layer, no extra bias unit */

  int        maxEpochs;
  int        maxHpSweeps;
  double     learningRate;                /*  (eta) -- initial learning rate */

  int        trainSubsetSize;             /*  No. images to be used in TRAINING the net */
  int        validSubsetSize;             /*  No. images to be used in VALIDATING the net */
  double     momentum;                    /*  (alpha) -- momentum coefficient */
  double     stopValidScore;              /*  Stop of validation score reaches this value (and early stop enabled)   */
  double     skipSubsetSize;              /*  Number of images to skip, to implement k-fold vlaidation; 0=disable    */
  int        updateRuleType;              /*  see psds.h for a list of defined values                                */
  int        minibatchSize;               /*  For minibatch updates: number of training cases per weight update      */
  double     noise;                       /*  Amount of noise to add to image for 2nd pass; 0.0 = disable            */
  int        regularizeType;              /*  What kind, if any, regularization to use to calculate output error     */
  double     weightDecay;                 /*  Weight decay per epoch; 0.00 = no weight decay. Try 0.00001 or lower   */
  double     kRegularize;                 /*  (lambda) -- regularization coefficient                                 */
  double     randomWtScale;               /*  Used for scaling random values loaded into each weight at init time    */
} hyperparam_t;


typedef struct _classInfo_t
{
  int     class1, class2;     /* First and second most likely class */
  double  conf;  /* Confidence */
} classInfo_t;


typedef struct _net_t
{
  hyperparam_t hyperparams;

  int    state;                             /* 0: training; 1: validation; 2: test */

  /* INPUT LAYER */
  double X[MAX_INPUT_LAYER_CNT];            /* Matrix of input layer values */
  int    label;                             /* Current training label */
#if( FB_IMP == FBI_SAS )
  int    specialFordwardUpdate[2];
#endif
  /* HIDDEN LAYER 0 */
  double W1[MAX_HIDDEN0_LAYER_CNT][MAX_INPUT_LAYER_CNT];
  double z2[MAX_HIDDEN0_LAYER_CNT];             /* Activaty of hidden (2nd) layer; weighted sum  */
  double a2[MAX_HIDDEN0_LAYER_CNT];             /* Hidden layer activations (output of sigmoid function) */
  double W1delta[MAX_HIDDEN0_LAYER_CNT][MAX_INPUT_LAYER_CNT];

  /* OUTPUT LAYER */
  double W2[MAX_OUTPUT_LAYER_CNT][MAX_HIDDEN0_LAYER_CNT];
  double z3[MAX_OUTPUT_LAYER_CNT];              /* Activity of output layer; weighted sum  */
  double Yhat[MAX_OUTPUT_LAYER_CNT];            /* Output layer activations (output of sigmoid function) */
  double Y[MAX_OUTPUT_LAYER_CNT];               /* Desired output layer activations (derived from training data) */
  double W2delta[MAX_OUTPUT_LAYER_CNT][MAX_HIDDEN0_LAYER_CNT];

  double outputLayerErrors[MAX_OUTPUT_LAYER_CNT];   /* Output layer error quantities */
  double outputErrorTotal;


  /* Partial derivatives? */
  double DeltaO[MAX_OUTPUT_LAYER_CNT];          /* Partial derivative? */
  double SumDOW[MAX_HIDDEN0_LAYER_CNT];         /* "Sum of DeltaO[] * W2[]"  (inside backprop calc loop)*/
  double DeltaH[MAX_HIDDEN0_LAYER_CNT];         /* ? (inside backprop) */

  double junkBuffer[100000];
#if( 0 )
  double eta;  /* DEPRECATED -- use hyperparams.learningRate */
  double alpha; /* DEPRECATRED -- use hyperparams.momentum */
#endif
} net_t;

typedef struct _imageData_t
{
  double  trainImg[TRAIN_FILE_SIZE][28*28];
  int     trainLabel[TRAIN_FILE_SIZE];
  double  testImg[TEST_FILE_SIZE][28*28];
} imageData_t;

#if( RELOAD_WEIGHTS )
#include "loadWeights.c"
#endif

/* PROTOTYPES */
void forward( net_t *net );
void backward( net_t *net );


/*********************************************************************************************************************/
/*********************************************************************************************************************/
double computeL2norm( net_t * net )
{
  int i, j;
  double L2 = 0.0;

  /* Calculate L2 norm (magnitude of weights as a single vector) */
  for( i=0; i<net->hyperparams.hidden0LayerCnt-1; i++ )
  {
    for( j=0; j<net->hyperparams.inputLayerCnt; j++ )
    {
      L2 += net->W1[i][j] * net->W1[i][j];
    }
  }
  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    for( j=0; j<net->hyperparams.hidden0LayerCnt-1; j++ )
    {
      L2 += net->W2[i][j] * net->W2[i][j];
    }
  }
  L2 = sqrt( L2 );
#if( 0 )
  printf( "L2norm:%.6lf  ", L2 );
#endif
  return L2;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
int loadHyperparmsFromFile( hyperparam_t *p )
{
}

/*********************************************************************************************************************/
/* randomWeigth() returns a pseudorandom value in the range +/- 1.0                                                  */
/*********************************************************************************************************************/
double randomWeight( void )
{
  double r;

  r = (double)rand();     /* set r = 0...RAND_MAX */
  r /= (double)RAND_MAX;  /* set r = 0...1 */
  r *= 2.0;               /* set r = 0...2 */
  r -= 1.0;               /* set r = -1 ... +1 */

  return r;
}

/*********************************************************************************************************************/
/* This is slow; google "fast sigmoid function"                                                                      */
/*********************************************************************************************************************/
double sigmoid( double x )
{
  return( 1.0 / ( 1.0 + exp( -x ) ) );
}

/*********************************************************************************************************************/
/* ReLU function:  f(x) = max(0,x)                                                                                   */
/* By the way: derivative of RELU: if x > 0, derivative is 1; 0 otherwise                                            */
/*********************************************************************************************************************/
double relu( double x )
{
  return x > 0.0 ? x : 0.0;
}

/*********************************************************************************************************************/
/* boundsCheckOK returns true (non-zero) of the difference between values a and b is less than epsilon.              */
/*********************************************************************************************************************/
int boundsCheckOK( double a, double b, double epsilon )
{
  double diff = a - b;
  if( diff > 0.0 )
    return( diff < epsilon );
  else
    return( diff > -epsilon );
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void logToErrorCSVFile( int i, double val, char mode[] )
{
  FILE *f = fopen( "errorLog.csv", mode );
  if( f )
  {
    fprintf( f, "%d, %lf\n", i, val );
    fclose( f );
  }
}

/*********************************************************************************************************************/
/* reloadWeights()                                                                                                   */
/*********************************************************************************************************************/
#if( RELOAD_WEIGHTS )
void reloadWeights( net_t *net )
{
  int i, j;

  for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
    for( j=0; j<net->hyperparams.inputLayerCnt; j++ )
      net->W1[i][j] = w1[i][j];

  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
    for( j=0; j<net->hyperparams.hidden0LayerCnt; j++ )
      net->W2[i][j] = w2[i][j];
}
#endif

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void dumpWeightsToSrcFile( net_t *net )
{
  int i, j, k;
  FILE *f;

  f = fopen( "loadWeights.c", "w" );
  if( NULL == f )
  {
    printf( "FAILED TO OPEN loadWeights.c for write\n" );
    goto dwthfCleanupAndExit;
  }

  /*---------*/
  /* DUMP W1 */
  /*---------*/
  fprintf( f, "static double w1[HIDDEN0_LAYER_CNT][INPUT_LAYER_CNT] =\n" );
  fprintf( f,
  "{\n" );
  for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
  {
    fprintf( f, "{ " );
    k = 0;
    for( j=0; j<net->hyperparams.inputLayerCnt; j++ )
    {
      fprintf( f, "%.9lf%s ", net->W1[i][j], j == net->hyperparams.inputLayerCnt-1 ? "" : "," );
      if( k++ >= 9 )
      {
        fprintf( f, "\n" );
        k = 0;
      }
    }
    fprintf( f, "}%s\n", i == net->hyperparams.hidden0LayerCnt-1 ? "" : "," );
  }
  fprintf( f, "};\n\n" );


  /*---------*/
  /* DUMP W2 */
  /*---------*/
  fprintf( f, "static double w2[OUTPUT_LAYER_CNT][HIDDEN0_LAYER_CNT] =\n" );
  fprintf( f,
  "{\n" );
  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    fprintf( f, "{ " );
    k = 0;
    for( j=0; j<net->hyperparams.hidden0LayerCnt; j++ )
    {
      fprintf( f, "%.9lf%s ", net->W2[i][j], j == net->hyperparams.hidden0LayerCnt-1 ? "" : "," );
      if( k++ >= 9 )
      {
        fprintf( f, "\n" );
        k = 0;
      }
    }
    fprintf( f, "}%s\n", i == net->hyperparams.outputLayerCnt-1 ? "" : "," );
  }
  fprintf( f, "};\n\n" );


dwthfCleanupAndExit:
  if( NULL != f )
    fclose( f );
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void dumpNetToHTML( net_t *net, char mode[] )
{
  int i, j;
  FILE *f = fopen( "net_dump.html", mode );
  if( f )
  {
    fprintf( f, "<TABLE BORDER=1 CELLPADDING=2 STYLE=\"font-family: arial; border: 1px solid black; "
                "border-collapse: collapse\">" );

    /*-------------*/
    /* INPUT LAYER */
    /*-------------*/
    for( i=0; i<net->hyperparams.inputLayerCnt; i++ )
    {
      fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d]</TD><TD>%lf</TD></TR>\n",
               i ? "" : "<B>X</B>", i,  net->X[i] );
    }

    fprintf( f, "<TR><TD COLSPAN=2 BGCOLOR=D0D0D0></TD></TR>\n" );

    /*----------------*/
    /* HIDDEN LAYER 0 */
    /*----------------*/
    for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
    {
      fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d]</TD><TD>%lf</TD></TR>\n",
               i ? "" : "<B>hiddenLayer0</B>", i,  net->a2[i] );
    }

    fprintf( f, "<TR><TD COLSPAN=2 BGCOLOR=D0D0D0></TD></TR>\n" );

    for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
      for( j=0; j<net->hyperparams.outputLayerCnt; j++ )
      {
        fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d][%d]</TD><TD>%lf</TD></TR>\n",
                 i+j ? "" : "W1", i, j, net->W1[i][j] );
      }

    fprintf( f, "<TR><TD COLSPAN=2 BGCOLOR=D0D0D0></TD></TR>\n" );

    for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
      for( j=0; j<net->hyperparams.outputLayerCnt; j++ )
      {
        fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d][%d]</TD><TD>%lf</TD></TR>\n",
                 i+j ? "" : "W1delta", i, j, net->W1delta[i][j] );
      }

    fprintf( f, "<TR><TD COLSPAN=2 BGCOLOR=D0D0D0></TD></TR>\n" );


    /*--------------*/
    /* OUTPUT LAYER */
    /*--------------*/
    for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
    {
      fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d]</TD><TD>%lf</TD></TR>\n",
               i ? "" : "<B>z3</B>", i, net->z3[i] );
    }

    fprintf( f, "<TR><TD COLSPAN=2 BGCOLOR=D0D0D0></TD></TR>\n" );

    for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
      for( j=0; j<net->hyperparams.outputLayerCnt; j++ )
      {
        fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d][%d]</TD><TD>%lf</TD></TR>\n",
                 i+j ? "" : "W2", i, j, net->W2[i][j] );
      }

    fprintf( f, "<TR><TD COLSPAN=2 BGCOLOR=D0D0D0></TD></TR>\n" );

    for( i=0; i<net->hyperparams.hidden0LayerCnt; i++ )
      for( j=0; j<net->hyperparams.outputLayerCnt; j++ )
      {
        fprintf( f, "<TR><TD ALIGN=RIGHT>%s [%d][%d]</TD><TD>%lf</TD></TR>\n",
                 i+j ? "" : "W2delta", i, j, net->W2delta[i][j] );
      }

    fprintf( f, "</TABLE><BR><BR>\n" );
    fclose( f );
  }
}

/*********************************************************************************************************************/
/* Dump the contents of the current input image vector                                                               */
/*********************************************************************************************************************/
void dumpInputLayerToHTML( hyperparam_t *hp, int label, double v[], char mode[] )
{
  int i, j, c;
  FILE *f = fopen( "X.html", mode );
  if( f )
  {
    fprintf( f, "<TABLE BORDER=1 STYLE=\"border: 1px solid black; border-collapse: collapse\">"
                "<TR><TD WIDTH=20>&nbsp;%d</TD><TD>", label );
    fprintf( f, "<TABLE CELLSPACING=0 BORDER=0><TR>\n" );
    j = 0;
    for( i=0; i<hp->inputLayerCnt-1; i++ )
    {
      c = (int)(v[i] * 255.0);
#if( 0 )
      /* RED */
/*      int color = 0xFFFFFF - (((int)v[i])<<8) - ((int)v[i]); */
      int color = 0xFFFFFF - ( c<<8) - c;
      fprintf( f, "<TD HEIGHT=3 WIDTH=1 BGCOLOR=%06x>", color );
#else
      /* BLACK */
      int color = 0xFFFFFF - (c | (c<<8) | (c<<16));
      fprintf( f, "<TD HEIGHT=3 WIDTH=1 BGCOLOR=%06x>", v[i] > 0.0 ? color : 0xF0F0F0 );
#endif
      if( ++j > 27 )
      {
        fprintf( f, "</TR>\n<TR>" );
        j=0;
      }
    }
    fprintf( f, "</TR></TABLE>\n" );
    fprintf( f, "</TD></TR></TABLE>\n" );
    fclose( f );
  }
  else
  {
    printf( "ERROR generating dump file!\n" );
  }
}

/*********************************************************************************************************************/
/* initNet() initializes all weights, and the values of all bias units.                                              */
/* When initializing weights, random values scalse by the number of a unit's inputs are used.                        */
/*********************************************************************************************************************/
void initNet( net_t *net )
{
  int i, j;

#if( FB_IMP == FBI_SAS )
  net->specialFordwardUpdate[0] = net->specialFordwardUpdate[1] = -1;
#endif

#if( !BACKPROP_DEV_MODE )

  /* Initialize the two bias unit values */
  net->X[net->hyperparams.inputLayerCnt-1] = net->a2[net->hyperparams.hidden0LayerCnt-1] = 1.0;

  /* Initialize hidden0 layer weights with random values */
  for( i=0; i<net->hyperparams.hidden0LayerCnt-1; i++ )
  {
    for( j=0; j<net->hyperparams.inputLayerCnt; j++ )
    {
#if( 0 )
      net->W1[i][j] = randomWeight() / sqrt( (double)net->hyperparams.inputLayerCnt );
#else
      net->W1[i][j] = randomWeight() * net->hyperparams.randomWtScale;
#endif
      net->W1delta[i][j] = 0.0;
    }
  }

  /* Initialize output layer weights with random values */
  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    for( j=0; j<net->hyperparams.hidden0LayerCnt; j++ )
    {
#if( 0 )
      net->W2[i][j] = randomWeight() / sqrt( (double)net->hyperparams.hidden0LayerCnt );
#else
      net->W2[i][j] = randomWeight() * net->hyperparams.randomWtScale;
#endif
      net->W2delta[i][j] = 0.0;
    }
  }

#else

    /* For use with   https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/   */
    net->X[0]              = 0.05;
    net->X[1]              = 0.10;

    net->W1[0][0]  = 0.15;
    net->W1[0][1]  = 0.20;
    net->W1[1][0]  = 0.25;
    net->W1[1][1]  = 0.30;

/*
    net->hiddenLayer0BiasWeights[0] = 0.35;
    net->hiddenLayer0BiasWeights[1] = 0.35;
*/

    net->W2[0][0]   = 0.40;
    net->W2[0][1]   = 0.45;
    net->W2[1][0]   = 0.50;
    net->W2[1][1]   = 0.55;

/*
    net->outputLayerBiasWeights[0]  = 0.60;
    net->outputLayerBiasWeights[1]  = 0.60;
*/
    net->Y[0]                    = 0.01;
    net->Y[1]                    = 0.99;
#endif
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void skipLineInFile( FILE *f )
{
  char line[MAX_LINE_SIZE];
  fgets( line, sizeof( line ), f ); /* Skip past the first line, which is just labels */
}

/*********************************************************************************************************************/
/* loadImageData() allocates and returns an image data structure containing all values from the specified files.     */
/*********************************************************************************************************************/
imageData_t * loadImageData( char trainFilePath[], char testFilePath[] )
{
  char  line[MAX_LINE_SIZE], valStr[16];
  int   len, i, j, k, id_idx = 0;
  FILE  *f = NULL;
  imageData_t *id = NULL;

  /* Allocate the network structure */
  id = malloc( sizeof( imageData_t ) );
  if( NULL == id )
  {
    printf( "FAILED to allocate memory for image data structure!\n" );
    goto cleanupAndExit;
  }

  /**-----------------------**/
  /**  LOAD TRAINING DATA   **/
  /**-----------------------**/
  f = fopen( trainFilePath, "r" );  /* Open training image file */
  if( NULL == f )
  {
    printf( "FAILED to open training data file!\n" );
    goto cleanupAndExit;
  }

  skipLineInFile( f );
  id_idx = 0;

  /* loop line-by-line */
  while( fgets( line, sizeof( line ), f ) != NULL ) /* Read one line (image) from the training set */
  {
    len = strlen( line );
    i=k=0;
    id->trainLabel[id_idx] = -1; /* We do this as a reminder we have not picked off the label value yet */

    /* Traverse the line (image) and process each pixel */
/*    while( (i < len) && (k < hp->inputLayerCnt-1) ) */
    while( i < len )
    {
      j=0; /* Index into valStr[] token buffer */
      /* Copy the substring up to the comma */
      while( (i < len) && ( line[i] != ',' ) )
      {
        valStr[j++] = line[i++];
      }
      valStr[j++] = 0; /* NULL-terminate */
      i++;  /* Skip past comma */
      /* If in training or validation mode, the first value in the line is the label (0...9); capture that */
      if( id->trainLabel[id_idx] < 0 ) /* We haven't grabbed the label already */
      {
        id->trainLabel[id_idx] = atoi( valStr );
      }
      else
      {
        id->trainImg[id_idx][k++] = ((double) atoi( valStr )) / 256.0; /* Normalize the pixel values as we load them */
      }
    }
    id_idx++;
  }
  fclose( f );

  /**-------------------**/
  /**  LOAD TEST DATA   **/
  /**-------------------**/
  f = fopen( testFilePath, "r" );  /* Open test image file */
  if( NULL == f )
  {
    printf( "FAILED to open test data file!\n" );
    goto cleanupAndExit;
  }

  skipLineInFile( f );
  id_idx = 0;

  /* loop line-by-line */
  while( fgets( line, sizeof( line ), f ) != NULL ) /* Read one line (image) from the training set */
  {
    len = strlen( line );
    i=k=0;

    /* Traverse the line (image) and process each pixel */
/*    while( (i < len) && (k < hp->inputLayerCnt-1) )  */
    while( i < len )
    {
      j=0; /* Index into valStr[] token buffer */
      /* Copy the substring up to the comma */
      while( (i < len) && ( line[i] != ',' ) )
      {
        valStr[j++] = line[i++];
      }
      valStr[j++] = 0; /* NULL-terminate */
      i++;  /* Skip past comma */
      id->testImg[id_idx][k++] = ((double) atoi( valStr )) / 256.0; /* Normalize the pixel values as we load them */
    }
    id_idx++;
  }
  fclose( f );

cleanupAndExit:
  return id;
}

/*********************************************************************************************************************/
/* getDigitClassification() finds the largest of the net output values, and returns its index                        */
/*********************************************************************************************************************/
void getDigitClassification( net_t *net, classInfo_t *ci )
{
  int i, maxi= -1, max2i = -2;
  double max = -1.0, max2 = -1.0;
  double avg = net->Yhat[0];
  double avgOfLosers = 0.00001;  /* precents div by zero later on */

  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    avg += net->Yhat[i];
    if( net->Yhat[i] > max )
    {
      /* First, give whatever big brother had previously to little brother */
      max2 = max;
      max2i = maxi;

      /* Big brother now takes this new, bigger one */
      max = net->Yhat[i];
      maxi = i;
    }
    else if( net->Yhat[i] > max2 )
    {
      max2 = net->Yhat[i];
      max2i = i;
    }
  }

  ci->class1 = maxi;
  ci->class2 = max2i;

  avgOfLosers += (avg - max) / (double)(net->hyperparams.outputLayerCnt-1);

#if( 1 )
  ci->conf = ((max-max2)*(max-max2))/avgOfLosers;
#else
  ci->conf = (max-avgOfLosers) * (max-avgOfLosers);
#endif
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
#if( FB_IMP == FBI_MM )
void backward( net_t *net )
{
}
#endif

/*********************************************************************************************************************/
/*********************************************************************************************************************/
#if( FB_IMP == FBI_JB )
void backward( net_t *net )
{
  int   i, j, k;

#if( DEBUG_LEVEL > 1 )
  printf( "backward() (JB implementation)\n" );
  fflush( stdout );
#endif

  /* Back-propagate errors to hidden layer */
  for( j = 0; j < net->hyperparams.hidden0LayerCnt; j++ )
  {
    net->SumDOW[j] = 0.0;
    for( k = 0; k < net->hyperparams.outputLayerCnt-1; k++ )
    {
      net->SumDOW[j] += net->W2[k][j] * net->DeltaO[k];
    }
    net->DeltaH[j] = net->SumDOW[j] * net->a2[j] * (1.0 - net->a2[j]);
  }

  /* Update hidden layer weights W1 */
  for( j = 0; j < net->hyperparams.hidden0LayerCnt-1; j++ )
  {
    /* Update bias weight, which is a different partial derivative calculation from the other weights */
    net->W1delta[j][net->hyperparams.inputLayerCnt-1] = ( net->hyperparams.learningRate * net->DeltaH[j] )
          + ( net->hyperparams.momentum * net->W1delta[j][net->hyperparams.inputLayerCnt-1] );
    net->W1[j][net->hyperparams.inputLayerCnt-1] += net->W1delta[j][net->hyperparams.inputLayerCnt-1];

    /* Update the other hidden layer weights */
    for( i = 0; i < net->hyperparams.inputLayerCnt-1; i++ )
    {
      net->W1delta[j][i] = ( net->hyperparams.learningRate * net->X[i] * net->DeltaH[j] )
                         + ( net->hyperparams.momentum * net->W1delta[j][i] );
      net->W1[j][i] += net->W1delta[j][i];
    }
  }

  /* Update output layer weights W2 */
  for( k = 0; k < net->hyperparams.outputLayerCnt-1; k++ )
  {
    /* Update bias weights */
    net->W2delta[k][net->hyperparams.hidden0LayerCnt-1] = ( net->hyperparams.learningRate * net->DeltaO[k] )
        + ( net->hyperparams.momentum * net->W2delta[k][net->hyperparams.hidden0LayerCnt-1] );
    net->W2[k][net->hyperparams.hidden0LayerCnt-1] += net->W2delta[k][net->hyperparams.hidden0LayerCnt-1];

    /* Update the rest of the weights */
    for( j = 0; j < net->hyperparams.hidden0LayerCnt-1; j++ )
    {
      net->W2delta[k][j] = ( net->hyperparams.learningRate * net->a2[j] * net->DeltaO[k] )
                         + ( net->hyperparams.momentum * net->W2delta[k][j] );
      net->W2[k][j] += net->W2delta[k][j];
    }
  }
}
#endif

/*********************************************************************************************************************/
/* forward_sas() evaluates the entire network, from inputs to outputs, using tree pruning with the assumption that   */
/* backward_sas() is also in use which uses weight tweaking rather than partial diravitive-based backprop.           */
/*********************************************************************************************************************/
#if( FB_IMP == FBI_SAS )
void forward( net_t *net )
{
  int i, j, k;
  double d;

  /* Calculate hidden layer activations */
  /* NOTE: we may here to calculate all hidden units, or just two; if only two hidden units have been */
  /* specified, then just update those, else update all. */
  if( net->specialFordwardUpdate[0] == -1 )
  {
    /* Update all */
    for( i=0; i<HIDDEN0_LAYER_CNT-1; i++ )  /* -1, as bias unit by definition has fixed activation (1.0) */
    {
      d = 0.0;
      for( j=0; j<INPUT_LAYER_CNT; j++ )
      {
        d += (net->X[j] * net->W1[i][j]);
      }
      net->z2[i] = d;
      net->a2[i] = sigmoid( d );
    }
  }
  else
  {
    /* Just update these two */
    for( k=0; k<2; k++ )
    {
      i = net->specialFordwardUpdate[k];
      if( i > -1 )
      {
        d = 0.0;
        for( j=0; j<INPUT_LAYER_CNT; j++ )
        {
          d += (net->X[j] * net->W1[i][j]);
        }
        net->z2[i] = d;
        net->a2[i] = sigmoid( d );
      }
    }
    net->specialFordwardUpdate[0] = net->specialFordwardUpdate[1] = -1;
  }

  /* Calculate output layer activations */
  for( i=0; i<OUTPUT_LAYER_CNT; i++ )
  {
    d = 0.0;
    for( j=0; j<HIDDEN0_LAYER_CNT; j++ )
    {
      d += (net->a2[j] * net->W2[i][j]);
    }
    net->z3[i] = d;
    net->Yhat[i] = sigmoid( d );
  }

  /* Calculate error for each output neuron using the squared error function and sum them to get the total error */
  net->outputErrorTotal = 0.0;
  for( i=0; i<OUTPUT_LAYER_CNT; i++ )
  {
    /* Cost function: = 1/2 * (desired - actual)^^2  */
    net->outputLayerErrors[i] = 0.5 * ( net->Y[i] - net->Yhat[i] ) * ( net->Y[i] - net->Yhat[i] );
    net->outputErrorTotal += net->outputLayerErrors[i];
  }
#if( DEBUG_LEVEL > 1 )
  printf( "forward_sas(); net->a2[HIDDEN0_LAYER_CNT-1] = %.3lf\n", net->a2[HIDDEN0_LAYER_CNT-1] );
  fflush( stdout );
#endif
}
#endif

/*********************************************************************************************************************/
/* forward() evaluates the entire network, from inputs to outputs                                                    */
/*********************************************************************************************************************/
#if( FB_IMP == FBI_JB )
void forward( net_t *net )
{
  int    i, j, k;
  double d, L2norm = 0.0; /* computeL2norm( net ); foobar */

  /* Calculate hidden layer activations */
  for( i=0; i<net->hyperparams.hidden0LayerCnt-1; i++ )  /* -1, as bias unit by definition has fixed activation (1.0) */
  {
    d = 0.0;
    for( j=0; j<net->hyperparams.inputLayerCnt; j++ )
    {
      d += (net->X[j] * net->W1[i][j]);
    }
    net->z2[i] = d;
    net->a2[i] = sigmoid( d );
  }

  /* Calculate output layer activations */
  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    d = 0.0;
    for( j=0; j<net->hyperparams.hidden0LayerCnt; j++ )
    {
      d += (net->a2[j] * net->W2[i][j]);
    }
    net->z3[i] = d;
    net->Yhat[i] = sigmoid( d );
  }

  /* Calculate error for each output neuron using the squared error function and sum them to get the total error */
  net->outputErrorTotal = 0.0;
  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    net->DeltaO[i] = (net->Y[i] - net->Yhat[i])
                   * net->Yhat[i] * ( 1.0 - net->Yhat[i] ); /* Sigmoidal Outputs, SSE */

    /* Cost function: = 1/2 * (desired - actual)^^2  */
#if( 0 )
    net->outputLayerErrors[i] = 0.5 *
                                  (
                                    ( ( net->Y[i] - net->Yhat[i] ) * ( net->Y[i] - net->Yhat[i] ) )
                                    + ( net->hyperparams.kRegularize * L2norm )
                                  );
#else
    net->outputLayerErrors[i] = 0.5 * ( net->Y[i] - net->Yhat[i] ) * ( net->Y[i] - net->Yhat[i] );
#endif
    net->outputErrorTotal += net->outputLayerErrors[i];
  }
#if( DEBUG_LEVEL > 1 )
  printf( "forward(); net->a2[net->hyperparams.hidden0LayerCnt-1] = %.3lf\n",
          net->a2[net->hyperparams.hidden0LayerCnt-1] );
  fflush( stdout );
#endif
}
#endif

/*********************************************************************************************************************/
/* forward() evaluates the entire network, from inputs to outputs                                                    */
/*********************************************************************************************************************/
#if( FB_IMP == FBI_DEFAULT || FB_IMP == FBI_MM )
void forward( net_t *net )
{
  int i, j, k;
  double d;

  /* Calculate hidden layer activations */
  for( i=0; i<HIDDEN0_LAYER_CNT-1; i++ )  /* -1, as bias unit by definition has fixed activation (1.0) */
  {
    d = 0.0;
    for( j=0; j<INPUT_LAYER_CNT; j++ )
    {
      d += (net->X[j] * net->W1[i][j]);
    }
    net->z2[i] = d;
    net->a2[i] = sigmoid( d );
  }

  /* Calculate output layer activations */
  for( i=0; i<OUTPUT_LAYER_CNT; i++ )
  {
    d = 0.0;
    for( j=0; j<HIDDEN0_LAYER_CNT; j++ )
    {
      d += (net->a2[j] * net->W2[i][j]);
    }
    net->z3[i] = d;
    net->Yhat[i] = sigmoid( d );
  }

  /* Calculate error for each output neuron using the squared error function and sum them to get the total error */
  net->outputErrorTotal = 0.0;
  for( i=0; i<OUTPUT_LAYER_CNT; i++ )
  {
    /* Cost function: = 1/2 * (desired - actual)^^2  */
    net->outputLayerErrors[i] = 0.5 * ( net->Y[i] - net->Yhat[i] )
                                    * ( net->Y[i] - net->Yhat[i] );
    net->outputErrorTotal += net->outputLayerErrors[i];
  }
#if( DEBUG_LEVEL > 1 )
  printf( "forward(); net->a2[HIDDEN0_LAYER_CNT-1] = %.3lf\n", net->a2[HIDDEN0_LAYER_CNT-1] );
  fflush( stdout );
#endif
}
#endif

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void delta3
(
  double *dest[]  /* Pointer to destination array */
)
{
}

/*********************************************************************************************************************/
/* backward_scw() -- perform backpropagation on a neural network                                                     */
/*                                                                                                                   */
/* Based on Stephen C. Welch "neural networks demystified" video tutorials                                           */
/*                                                                                                                   */
/* Organization:                                                                                                     */
/*  X: input training data matrix (row count: # of input neurons; col count: # of data elements in each sample)      */
/*  W(¹): hidden layer weight matrix (row count: # neurons in that layer; col count: # if inputs for each neuron     */
/*        in that layer)                                                                                             */
/*        (one row for each example, one comn for each hidden unit)                                                  */
/*        ( w(¹)* is "w one transpose")                                                                              */
/*  z(²): activity of second layer                                                                                   */
/*  y: training label matrix                                                                                         */
/*  ŷ: network's estimate of y                                                                                       */
/*  z: sum of weighted inputs                                                                                        */
/*  a: activation, output of a neuron (output of sigmoid activation function)                                        */
/*                                                                                                                   */
/* Equations from video part 2, 4:00                                                                                 */
/*  z(²) = X · W(¹)                     ... ("z 2 equals X times W one")   (1)                                       */
/*  a(²) = ƒ( z(²) )                                                       (2)                                       */
/*  z(³) = a(²) · W(²)                                                     (3)                                       */
/*     ŷ = ƒ( z(³) )                                                       (4)                                       */
/*     J = Σ ½(y - ŷ)²                  ... cost function                  (5)                                       */
/*                                                                                                                   */
/*  ƒ(z) = 1.0 / (1.0 + exp(-z))        ... sigmoid function                                                         */
/*  ƒ´(z) = exp(-z) / (1.0 + exp(-z))²  ... [video part 4, 2:50]                                                     */
/*                                                                                                                   */
/* Combined cost function equation - evaluating entire network [video part 3, 4:21]                                  */
/*     J = Σ ½(y - ƒ( ƒ( X·W(¹) ) · W(²) ) )²                                                                        */
/*                                                                                                                   */
/*      ∂J/∂W : rate of change of J with respect to a change in one weight value in W                                */
/*********************************************************************************************************************/
#if( BACKPROP_IMP == FBI_SCW )
void backward( net_t *net )
{
  /*---------------------------------------------------------------------------------------------*/
  /* OUTPUT LAYER BACKPROPAGATION                                                                */
  /* ∂J/∂W(²) =  -(y-ŷ) · ƒ´( z(³))  · ( ∂z(³)/∂W(²) )                   [video part 4, 5:00]    */
  /*          = (-(y-ŷ) · ƒ´( z(³))) · ( ∂z(³)/∂W(²) )                                           */
  /*          =  𝛿(³)                · ( ∂z(³)/∂W(²) )                                           */
  /*                                                                                             */
  /* ("delta three: back prop error function")                           [video part 4, 4:36]    */
  /*---------------------------------------------------------------------------------------------*/


  /*---------------------------------------------------------------------------------------------*/
  /* HIDDEN LAYER 0 BACKPROPAGATION                                                              */
  /* ∂J/∂W(¹) = 𝛿(³) · (∂z(³)/∂a(²)) · (∂a(²)/∂W(¹))                    [video part 4, 6:00]     */
  /*          = 𝛿(³) ·    W(²)*      · (∂a(²)/∂W(¹))                    [video part 4, 6:20]     */
  /*          = 𝛿(³) ·    W(²)*      · (∂z(²)/∂z(²)) · (∂z(²)/∂W(¹))    [video part 4, 6:24]     */
  /*          = 𝛿(³) ·    W(²)*      · ƒ´(z(²))      · (∂z(²)/∂W(¹))    [video part 4, 6:31]     */
  /*          = 𝛿(³) ·    W(²)*      · ƒ´(z(²))      · X*               [video part 4, 6:59]     */
  /*          = X* ·   𝛿(³) · W(²)* · ƒ´(z(²))                          (eq reordered)           */
  /*          = X* · ( 𝛿(³) · W(²)* · ƒ´(z(²)) )                        simplify terms           */
  /*          = X* · 𝛿(²)                                               [video part 4, 6:59]     */
  /*---------------------------------------------------------------------------------------------*/
}
#endif

/*********************************************************************************************************************/
/*********************************************************************************************************************/
#if( FB_IMP == FBI_SAS )
void backward( net_t *net )
{
  int    i, j, k, done;
  double E;
  double olwg[OUTPUT_LAYER_CNT][HIDDEN0_LAYER_CNT]; /* OUTPUT LAYER GRADIENTS*/
  double hlwg[HIDDEN0_LAYER_CNT][INPUT_LAYER_CNT]; /* HIDDEN LAYER GRADIENTS */
  double olbg[OUTPUT_LAYER_CNT];  /* OUTPUT LAYER BIAS GRADIENTS */
  double hlbg[HIDDEN0_LAYER_CNT];  /* HIDDEN LAYER BIAS GRADIENTS */
  double d, dd;
  double tweak = WEIGHT_TWEAK;

  /* Capture baseline error prior to any weight perterbations */
  E = net->outputErrorTotal;

  /*------------------------------------------------------------*/
  /* Perturb each OUTPUT LAYER weight and capture each gradient */
  /*------------------------------------------------------------*/
  for( j=0; j<HIDDEN0_LAYER_CNT; j++ ) /* For each HIDDEN LAYER node... */
  {
    for( i=0; i<OUTPUT_LAYER_CNT; i++ ) /* for each OUTPUT LAYER node... */
    {
      d = net->W2[i][j]; /* Temporarily save weight */
      net->W2[i][j] += tweak; /* Perturb the weight */
      forward( net ); /* Recalculate error */
      olwg[i][j] = net->outputErrorTotal - E; /* Capture gradient resulting from one weight tweak */
      net->W2[i][j] = d; /* Restore the weight */
    }
  }

  /*------------------------------------------------------------*/
  /* Perturb each HIDDEN LAYER weight and capture each gradient */
  /*------------------------------------------------------------*/
  for( i=0; i<HIDDEN0_LAYER_CNT-1; i++ ) /* for each HIDDEN LAYER node... */
  {
    /* If we have moved to a subsequent hidden unit, clean up the previous one  */
    if( i > 0 )
      net->specialFordwardUpdate[1] = i-1;
    for( j=0; j<INPUT_LAYER_CNT; j++ ) /* For each INPUT LAYER node... */
    {
      d = net->W1[i][j]; /* Temporarily save weight */
      net->W1[i][j] += tweak;
      net->specialFordwardUpdate[0] = i; /* Just update the hidden unit associated with this weight */
      forward( net );
      hlwg[i][j] = net->outputErrorTotal - E; /* Capture gradient resulting from one weight tweak */
      net->W1[i][j] = d; /* Restore the weight */
    }
  }

  /*-----------------------------------------------------------------------------------------------------------*/
  /* Now that we have a complete set of gradients, we will apply them iteratively, without recalculating them, */
  /* until the network output error is no longer dropping.                                                     */
  /*-----------------------------------------------------------------------------------------------------------*/
  done = k = 0;
  E = net->outputErrorTotal; /* Probably not necessary, but cheap insurance */
  while( !done )
  {
    /*---------------------------------------*/
    /* Apply all OUTPUT LAYER weight changes */
    /*---------------------------------------*/
    for( i=0; i<OUTPUT_LAYER_CNT; i++ ) /* for each OUTPUT LAYER node... */
    {
      for( j=0; j<HIDDEN0_LAYER_CNT; j++ ) /* For each HIDDEN LAYER node... */
      {
        net->W2[i][j] -= olwg[i][j];
      }
    }

    /*---------------------------------------*/
    /* Apply all HIDDEN LAYER weight changes */
    /*---------------------------------------*/
    for( i=0; i<HIDDEN0_LAYER_CNT-1; i++ ) /* for each HIDDEN LAYER node... */
    {
      for( j=0; j<INPUT_LAYER_CNT; j++ ) /* For each INPUT LAYER node... */
      {
        net->W1[i][j] -= hlwg[i][j];
      }
    }

    /* Check the error after updating the weights with the current set of gradients, quit of error rising */
    forward( net );
#if( 0 )
    printf( "grad iter %d; E = %.9lf  Enet = %.9lf \n", k, E, net->outputErrorTotal );
#endif
    k++;
    if( net->outputErrorTotal > E )
      done = 1;
    if( k > RUN_DWN_GRAD_LMT )
      done = 1;
  } /* end while (!done ) */
}
#endif

/*********************************************************************************************************************/
/* based on Matt Mazur tutorial                                                                                      */
/*********************************************************************************************************************/
#if( BACKPROP_IMP == FBI_MM )
void backward( net_t *net )
{
  int    i, ii, j;

  /* partial derivatives */
  double d1, d2, d3, d4, d5, d6, d7, d8, e;
  double cell_f28, cell_b9, cell_w30, cell_w31, cell_w32, cell_w33;
  double cell_q28, cell_q29, cell_q30, cell_q31, cell_q32, cell_q33;

  /*--------------*/
  /* OUTPUT LAYER */
  /*--------------*/
  /* Calculate the partial derivative of the total error w.r.t. each output weight */
  for( j=0; j<HIDDEN0_LAYER_CNT; j++ ) /* For each HIDDEN LAYER node... */
  {
    for( i=0; i<OUTPUT_LAYER_CNT; i++ ) /* for each OUTPUT LAYER node... */
    {
      /* Calculate ∂ of total error w.r.t. each output layer weight */
      d1 = -( net->Y[i] - net->Yhat[i] );   /* "∂E_total / out_o[i]" */
      d2 = ( net->Yhat[i] * (1.0 - net->Yhat[i] ) );  /* "∂out_o[i] / net_o[i]" */
      d3 = net->a2[j];                                    /* "∂net_o[i] / w[j]" */
      e = d1 * d2 * d3;

      /* Store the amount we will later modify the output layer weights. */
      net->W2delta[i][j] = e * net->learningRate;
#if( 0 )
      printf( "backprop, output layer:  d1:%lf  d2:%lf  d3:%lf  e:%lf\n", d1, d2, d3, e );
#endif
    }
  }

  /*----------------*/
  /* HIDDEN LAYER 0 */
  /*----------------*/
  for( j=0; j<HIDDEN0_LAYER_CNT; j++ )
  {
    for( i=0; i<OUTPUT_LAYER_CNT; i++ )
    {
#if( 0 )
      /* JUST RECALC EVERYTHING AGAIN, HERE; LATER, SAVE VALUES CALCULATED ABOVE AND REUSE HERE */
      /* Calculate pd of total error w.r.t. each hidden layer 0 weight  */

      /* CELL_R7 = -(CELL_O3 - CELL_L3) */
      d1 = -( net->Yhat[i] - net->Y[i] );   /* "∂E_total / ∂out_o1"  [CELL_R7]  */

      /* CELL_R8 = CELL_L3 * ( 1.0 - CELL_L3 ) */
      d2 = ( net->Y[i] * (1.0 - net->Y[i] ) );  /* "∂out_o1  / ∂net_o1"  [CELL_R8]  */

      d3 = d1 * d2;                                                 /* "∂E_o1    / ∂net_o1"  [CELL_Q28] */
      d4 = net->W2[i][j];                                           /* "∂net_o1  / ∂out_h1"  [CELL_Q29] */
      d5 = d3 * d4;                                                 /* "∂E_o1    / ∂out_h1"  [CELL_Q30] */
      d6 = 0.0;                                                     /* "" [CELL_Q31] */
      d7 = 0.0;                                                     /* "" [CELL_Q32] */
      d8 = d6 * d7;                                                 /* "" [CELL_Q33] */


      cell_w30 = cell_q30 + cell_q33;                                 /* "∂E_total / ∂out_h1" [CELL_W30] */

   /* CELL_W31 = CELL_G3 * ( 1.0 - CELL_G3) */
      cell_w31 = cell_g3 * (1.0 - cell_g3 );                           /* cell_31: "∂out_h1 / ∂net_h1" */

      /* CELL_W32 = "i1" */
      cell_w32 = net->inputs[foobar];  /* cell_w32: "∂net_h1 / ∂w_1" */

      /* CELL_W33 = CELL_W30 * CELL_W31 * CELL_W32 */
      cell_w33 = cell_w30 * cell_w31 * cell_w32;                      /* cell_w33: "∂E_total / ∂w_1" */

      /* Calculate and store hidden weight 0 deltas */
      /* CELL_F28 = CELL_B9 * CELL_W33 */
      net->W1delta[i][j] = e * net->learningRate;
#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#if( 0 )
      ddd = net->W1[i][j]; /* [] */

      E_o1__wrt__out_o1 = -( net->Yhat[i] - net->Y[i] );

      E_o1__wrt__net_o1 = E_o1__wrt__out_o1 * out_o1__wrt__net_o1;
      E_o1__wrt__out_h1 = E_o1__wrt__net_o1 * net_o1__wrt__out_h1;

      E_o2__wrt__out_h1 = 
      E_total__wrt__out_h1 = E_o1__wrt__out_h1 + E_o2__wrt__out_h1;


      d2 = ( net->Y[i] * (1.0 - net->Y[i] ) );  /* "PD out_h1  w.r.t. net_h1"  */
      d3 = net->W2[i][j];                           /* "PD net_h1  w.r.t. out_h1"  */
      printf( "d2: %lf  d3:%lf \n", d2, d3 );

#endif
#if( 0 )
      printf( "backprop, hidden 0 layer: d1:%lf  d2:%lf  d3:%lf\n", d1, d2, d3  );
#endif
    }
  }

  /*--------------------------------------------------------------------------*/
  /* Modify all network weights according to the deltas previously calculated */
  /*--------------------------------------------------------------------------*/
  for( i=0; i<OUTPUT_LAYER_CNT; i++ )
  {
    for( j=0; j<HIDDEN0_LAYER_CNT; j++ )
    {
      net->W2[i][j] -= net->W2delta[i][j];
      net->W1[i][j] -= net->W1delta[i][j];
    }
  }
}
#endif

/*********************************************************************************************************************/
/*********************************************************************************************************************/
#if( FB_IMP == FBI_MM )
void doUnitTest( net_t *net )
{
  const double maxErr = 0.00001;

  forward( net );

  dumpNetToHTML( net, "w" );

  printf( "** FEEDFORWARD TEST **\n" );

  printf( "            Hidden0 node 0: %s\n",
          boundsCheckOK( net->a2[0],          0.593269992, maxErr ) ? "OK":"*FAIL*" );

  printf( "            Hidden0 node 1: %s\n",
          boundsCheckOK( net->a2[1],          0.596884378, maxErr ) ? "OK":"*FAIL*" );

  printf( "             output node 0: %s\n",
          boundsCheckOK( net->Yhat[0],           0.751365070, maxErr ) ? "OK":"*FAIL*" );

  printf( "             output node 1: %s\n",
          boundsCheckOK( net->Yhat[1],           0.772928465, maxErr ) ? "OK":"*FAIL*" );

  printf( "          outputErrorTotal: %s\n",
          boundsCheckOK( net->outputErrorTotal,         0.298371109, maxErr ) ? "OK":"*FAIL*" );

  backward( net );

  dumpNetToHTML( net, "a" );

  printf( "** BACKPROP TEST **\n" );

  printf( "  W2[0][0]: %s\n",
          boundsCheckOK(  net->W2[0][0], 0.358916480, maxErr ) ? "OK":"*FAIL*" );

  printf( "  W2[0][1]: %s\n",
          boundsCheckOK(  net->W2[0][1], 0.408666186, maxErr ) ? "OK":"*FAIL*" );

  printf( "  W2[1][0]: %s\n",
          boundsCheckOK(  net->W2[1][0], 0.511301270, maxErr ) ? "OK":"*FAIL*" );

  printf( "  W2[1][1]: %s\n",
          boundsCheckOK(  net->W2[1][1], 0.561370121, maxErr ) ? "OK":"*FAIL*" );


  printf( "  W1[0][0]: %s\n",
          boundsCheckOK( net->W1[0][0], 0.149780716, maxErr ) ? "OK":"*FAIL*" );

  printf( "  W1[0][1]: %s\n",
          boundsCheckOK( net->W1[0][1], 0.199561432, maxErr ) ? "OK":"*FAIL*" );

  printf( "  W1[1][0]: %s\n",
          boundsCheckOK( net->W1[1][0], 0.249751144, maxErr ) ? "OK":"*FAIL*" );

  printf( "  W1[1][1]: %s\n",
          boundsCheckOK( net->W1[1][1], 0.299502287, maxErr ) ? "OK":"*FAIL*" );
}
#endif

/*********************************************************************************************************************/
/* speedTest() -- math speed test. On TIGGER, takes 31 seconds as MUL double, result:  1.001000500117                */
/*                               takes 31 seconds as MUL float,  result:  1.001192092896                             */
/*                               takes 20 seconds as MUL int (same for long)                                         */
/*                               takes 21 seconds as ADD int (same for long)                                         */
/*********************************************************************************************************************/
void speedTest( void )
{
  int         i, j, iSpeed;
  float       fSpeed;
  double      dSpeed, secs;
  long double ldSpeed;
  clock_t     t1, t2;
  printf( "** math speed test **\n" );

  printf( "long double (%d bit) mul test...  ", (int)(sizeof(long double )*8) );
  fflush( stdout );
  ldSpeed = 1.0;
  t1 = clock(); /* Capture CPU time used thus far by our process */
  for( i=0; i<1000000000; i++ )
    ldSpeed *= 1.000001;
  t2 = clock();
  secs = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
  printf( "%.3lf Mflops\n", (1.0e09/secs)/1.0e6 );


  printf( "double (%d bit) mul test...  ", (int)(sizeof(double )*8) );
  fflush( stdout );
  dSpeed = 1.0;
  t1 = clock(); /* Capture CPU time used thus far by our process */
  for( i=0; i<1000000000; i++ )
    dSpeed *= 1.000001;
  t2 = clock();
  secs = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
  printf( "%.3lf Mflops\n", (1.0e09/secs)/1.0e6 );


  printf( "double (%d bit) div test...  ", (int)(sizeof(double )*8) );
  fflush( stdout );
  dSpeed = 1.0;
  t1 = clock(); /* Capture CPU time used thus far by our process */
  for( i=0; i<1000000000; i++ )
    dSpeed /= 0.999999;
  t2 = clock();
  secs = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
  printf( "%.3lf Mflops\n", (1.0e09/secs)/1.0e6 );


  printf( "float  (%d bit) mul test...  ", (int)(sizeof(float)*8) );
  fflush( stdout );
  fSpeed = 1.0;
  t1 = clock(); /* Capture CPU time used thus far by our process */
  for( i=0; i<1000000000; i++ )
    fSpeed *= 1.000001;
  t2 = clock();
  secs = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
  printf( "%.3lf Mflops\n", (1.0e09/secs)/1.0e6 );


  printf( "float  (%d bit) div test...  ", (int)(sizeof(float)*8) );
  fflush( stdout );
  fSpeed = 1.0;
  t1 = clock(); /* Capture CPU time used thus far by our process */
  for( i=0; i<1000000000; i++ )
    fSpeed /= 0.999999;
  t2 = clock();
  secs = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
  printf( "%.3lf Mflops\n", (1.0e09/secs)/1.0e6 );
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void  testAndGenerateSubmissionFile( net_t *net, imageData_t *imageData )
{
  int i;
  classInfo_t  classInfo;
  FILE *imageFile = NULL, *outputFile = NULL;
  char filePath[256];
  unsigned int timeStamp = time( NULL );
  int id_idx;

  /**----------------------------**/
  /**  GENERATE SUBMISSION FILE  **/
  /**----------------------------**/

  /* Generate timestamped submission file name */
  sprintf( filePath, "submission_%d.csv", timeStamp );

  /* Open submission file */
  outputFile = fopen( filePath, "w" );
  if( NULL == outputFile )
  {
    printf( "FAILED to open submission file!\n\n" );
    goto cleanupAndExit;
  }

  fprintf( outputFile, "ImageId,Label\n" );

  i = 1;

  for( id_idx = 0; id_idx < TEST_FILE_SIZE; id_idx++ )
  {
    if( !(i%100) )
      printf( "  classifying test image %d  \r", i );

    /* Load one test image into the net inputs */
    for( i=0; i<net->hyperparams.inputLayerCnt-1; i++ ) /* -1, so we don't overwrite bias value */
      net->X[i] = imageData->testImg[id_idx][i];

    forward( net ); /* Classify the current test image */
    getDigitClassification( net, &classInfo );
    fprintf( outputFile, "%d,%d\n", id_idx+1, classInfo.class1 );
  }
  fclose( outputFile );
  outputFile = NULL;


  /**--------------------------**/
  /**  GENERATE METADATA FILE  **/
  /**--------------------------**/
  /* Generate timestamped metadata file name and create the file */
  sprintf( filePath, "submission_%d.html", timeStamp );
  outputFile = fopen( filePath, "w" );
  if( NULL == outputFile )
  {
    printf( "Unable to open submission metadata file for write!\n" );
    goto cleanupAndExit;
  }

  fprintf( outputFile, "<HTML>\n" );
  fprintf( outputFile, "This metadata file corresponds to submisson file 'submission_%d.csv'<BR><BR>\n\n", timeStamp );
  fprintf( outputFile, "PS Deep Sandbox, built %s, %s<BR><BR>\n\n", __DATE__, __TIME__ );

  fprintf( outputFile, "<TABLE BORDER=1 CELLPADDING=2 STYLE=\"font-family: arial; border: 1px solid black; "
                       "border-collapse: collapse\">" );

  fprintf( outputFile, "<TR><TD COLSPAN=2 ALIGN=CENTER><B>Hyperparameters & config</B></TD></TR>\n" );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT WIDTH=500>Hidden layer 0 count</TD>"
                           "<TD ALIGN=CENTER WIDTH=400>%d</TD></TR>\n",    net->hyperparams.hidden0LayerCnt );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Max hyperparameter sweeps</TD>"
                           "<TD ALIGN=CENTER>%d</TD></TR>\n",    net->hyperparams.maxHpSweeps );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Max epochs</TD>"
                           "<TD ALIGN=CENTER>%d</TD></TR>\n",    net->hyperparams.maxEpochs );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Training subset size</TD>"
                           "<TD ALIGN=CENTER>%d</TD></TR>\n",    net->hyperparams.trainSubsetSize );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Validation subset size</TD>"
                           "<TD ALIGN=CENTER>%d</TD></TR>\n",    net->hyperparams.validSubsetSize );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Learning rate (\"eta\")</TD>"
                           "<TD ALIGN=CENTER>%.8lf</TD></TR>\n", net->hyperparams.learningRate );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Momentum (\"alpha\")</TD>"
                           "<TD ALIGN=CENTER>%lf</TD></TR>\n",   net->hyperparams.momentum );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Update rule type</TD><TD ALIGN=CENTER>" );
  switch( net->hyperparams.updateRuleType )
  {
    case UR_NOT_DEF:  fprintf( outputFile, "<not defined>" );                break;
    case UR_ONLINE:   fprintf( outputFile, "Online"        );                break;
    case UR_MBGD:     fprintf( outputFile, "Minibatch gradient descent"  );  break;
    default:          fprintf( outputFile, "?"  );                           break;
  }
  fprintf( outputFile, "</TD></TR>\n" );

  if( net->hyperparams.updateRuleType == UR_MBGD )
  {
    fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Minibatch size</TD>"
                             "<TD ALIGN=CENTER>%d</TD></TR>\n",    net->hyperparams.minibatchSize );
  }

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Noise</TD>"
                           "<TD ALIGN=CENTER>%.4lf</TD></TR>\n",    net->hyperparams.noise );

  fprintf( outputFile, "<TR><TD ALIGN=RIGHT>Weight decay</TD>"
                           "<TD ALIGN=CENTER>%.8lf</TD></TR>\n",    net->hyperparams.weightDecay );
  fprintf( outputFile, "</TABLE\n" );
  fprintf( outputFile, "</HTML>\n" );
  fclose( outputFile );
  outputFile = NULL;

cleanupAndExit:
  if( imageFile )
  {
    fclose( imageFile );
    imageFile = NULL;
  }
  if( outputFile )
  {
    fclose( outputFile );
    outputFile = NULL;
  }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void addNoise( net_t *net )
{
  int i;
  double r;

  for( i=0; i<net->hyperparams.inputLayerCnt-1; i++ )
  {
    r = (double)rand();     /* set r = 0...RAND_MAX */
    r /= (double)RAND_MAX;  /* set r = 0...1 */
    r *= 2.0;               /* set r = 0...2 */
    r -= 1.0;               /* set r = -1 ... +1 */
    net->X[i] += r * net->hyperparams.noise;
    if( net->X[i] <  0.0 ) net->X[i] = 0.0;
    if( net->X[i] >  1.0 ) net->X[i] = 1.0;
  }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void weightDecay( net_t * net, double decay )
{
  int i, j;
  double d;

  /* Perform weight decay regularization of hidden0 layer weights */
  for( i=0; i<net->hyperparams.hidden0LayerCnt-1; i++ )
  {
    for( j=0; j<net->hyperparams.inputLayerCnt; j++ )
    {
      d = net->W1[i][j] * net->W1[i][j] * decay;
      if( net->W1[i][j] > 0.0 )
        net->W1[i][j] -= d;
      else
        net->W1[i][j] += d;
/*      net->W1[i][j] *= ( 1.0 - decay );   */
    }
  }
  /* Perform weight decay regularization of output layer weights */
  for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
  {
    for( j=0; j<net->hyperparams.hidden0LayerCnt; j++ )
    {
      d = net->W2[i][j] * net->W2[i][j] * decay;
      if( net->W2[i][j] > 0.0 )
        net->W2[i][j] -= d;
      else
        net->W2[i][j] += d;
 /*     net->W2[i][j] *= ( 1.0 - decay ); */
    }
  }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void initHyperparamsFromDefaults( hyperparam_t *hp )
{
  /* Just to be safe, in case we missed anything below, zero out the entir ememory range */
  memset( (void *)hp, 0, sizeof( hyperparam_t ) );

  hp->inputLayerCnt     = INPUT_LAYER_CNT;
  hp->hidden0LayerCnt   = HIDDEN0_LAYER_CNT;
  hp->outputLayerCnt    = OUTPUT_LAYER_CNT;

  hp->maxEpochs         = MAX_EPOCHS;
  hp->maxHpSweeps       = MAX_HP_SWEEPS;
  hp->learningRate      = LEARNING_RATE;

  hp->trainSubsetSize   = TRAIN_SUBSET_SIZE;
  hp->validSubsetSize   = VALID_SUBSET_SIZE;
  hp->momentum          = MOMENTUM;
  hp->stopValidScore    = STOP_VALID_SCORE;
  hp->skipSubsetSize    = SKIP_SUBSET_SIZE;
  hp->updateRuleType    = UPDATE_RULE_TYPE;
  hp->minibatchSize     = MINIBATCH_SIZE;
  hp->noise             = NOISE;
  hp->regularizeType    = REGULARIZE_TYPE;
  hp->weightDecay       = WEIGHT_DECAY;
  hp->kRegularize       = K_REGULARIZE;
  hp->randomWtScale     = RANDOM_WT_SCALE;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
int main( int argc, char *argv[] )
{
  int           i, j, k, imageCount, id_idx, imageDumpCount, epoch, hpSweep, status = 0;
  clock_t       start, end, cpuTimeUsed, cpuDays, cpuHours, cpuMins, cpuSecs;
  double        d;
  int           currentTrainingLabel;
  net_t        *net = NULL;
  imageData_t  *imageData = NULL;
  double        trainGraphX[1000], trainGraphY[1000], validGraphX[1000], validGraphY[1000];
  double        bestValidationScore = 0.0;
  classInfo_t   classInfo;
  double        foo, bar;
  int           fooCnt, barCnt;
  hyperparam_t  hparams;

  /* Hyperparameter sweep variables */
  double        hpSweepGraphX[1000], hpSweepGraphY[1000];
  int           hpSweepGraphIdx = 0;
  int           hpSweepDone = 0;  /* Done with outermost parameter sweep look? */

  start = clock(); /* Capture beginning time */

  setpriority( PRIO_PROCESS, 0, PROCESS_PRIO );

  printf( "\n╔═══════════════════════════════╗\n" );
  printf(   "║  PS Deep Sandbox              ║\n" );
  printf(   "║  built %s, %s  ║\n", __DATE__, __TIME__ );
  printf(   "╚═══════════════════════════════╝\n" );

#if ( 0 )
  speedTest();
  goto cleanupAndExit;
#endif

  initHyperparamsFromDefaults( &hparams );

  /* Display hyperparameters */
  printf( "** Hyperparameters **\n" );
  printf( "  Hidden 0 Layer Count : %d\n",    hparams.hidden0LayerCnt );
  printf( " Max hyperparam sweeps : %d\n",    hparams.maxHpSweeps     );
  printf( "             Max epoch : %d\n",    hparams.maxEpochs       );
  printf( "  Training subset size : %d\n",    hparams.trainSubsetSize );
  printf( " Validaton subset size : %d\n",    hparams.validSubsetSize );
  printf( "      Stop valid score : %.6lf\n", hparams.stopValidScore  );
  printf( "        RELOAD_WEIGHTS : %d\n",    RELOAD_WEIGHTS              );
  printf( "\n" );

  /* getchar() blocks waiting for a character */
  /* getc( stdin ) also blocks waiting for a character */

#if( DEBUG_LEVEL > 1 )
  printf( "Size of net_t: %ld bytes\n", sizeof( net_t ) );
  fflush( stdout );
#endif

#if( DEBUG_LEVEL > 1 )
  printf( "About to allocate net...\n" );
  fflush( stdout );
#endif

  /* Create and initialize the network */
  net = malloc( sizeof( net_t ) );
  if( NULL == net )
  {
    printf( "FATAL: mem alloc failed (net)\n" );
    goto cleanupAndExit;
  }
  memcpy( &net->hyperparams, &hparams, sizeof( hparams ) );
  initNet( net ); /* Initialize the network */

#if( BACKPROP_DEV_MODE )
  printf( "** BACKPROP_DEV_MODE enabled **\n" );
  doUnitTest( net );
  goto cleanupAndExit;
#endif

#if( RELOAD_WEIGHTS )
  printf( "** reloading previously saved weights **\n" );
  reloadWeights( net );
#endif

  /* Load the entire image database into memory */
  printf( "Loading training and test image data\n" );
  imageData = loadImageData( TRAIN_FILE_PATH, TEST_FILE_PATH );
  if( NULL == imageData )
  {
    printf( "FAILED to load image data!\n" );
    goto cleanupAndExit;
  }
  printf( "All training and test image data loaded\n" );

  printf( "** training **\n" );

  /**------------------------------------------------------------------------------------------**/
  /** HYPERPARAMETER SWEEP LOOP  **  HYPERPARAMETER SWEEP LOOP  **  HYPERPARAMETER SWEEP LOOP  **/
  /**------------------------------------------------------------------------------------------**/
  hpSweep = 0;
  while( hpSweep < net->hyperparams.maxHpSweeps )
  {
#if( 0 )
srand( 10 );  /* Initialize pseudorandom number generator */
#endif
#if( 0 )
srand( hpSweep );
#endif
#if( 1 )
  srand( 12345 );  /* Initialize pseudorandom number generator */
#endif
#if( 0 )
  srand((int)time(NULL));  /* Initialize pseudorandom number generator */
#endif

    epoch = 0;
    imageDumpCount = 0;
    initNet( net );

    if( net->hyperparams.maxHpSweeps > 1 )
    {
      printf( "Hyperparameter sweep %d / %d max\n", hpSweep, net->hyperparams.maxHpSweeps );
    }

    /**--------------------------------------------------------------------------**/
    /**  TRAINING LOOP  **  TRAINING LOOP  **  TRAINING LOOP  **  TRAINING LOOP  **/
    /**--------------------------------------------------------------------------**/
    id_idx = 0;
    net->state = 0; /* Its training time! */
    while( epoch < net->hyperparams.maxEpochs )
    {
      /**------------**/
      /**  TRAINING  **/
      /**------------**/
      /* Loop across images */
      for( imageCount = 0; imageCount < TRAIN_SUBSET_SIZE; imageCount++ )
      {
        /* Load an image & label from the data set into the net inputs */
        for( i=0; i<net->hyperparams.inputLayerCnt-1; i++ )
          net->X[i] = imageData->trainImg[id_idx][i];
        net->label = imageData->trainLabel[id_idx];
        id_idx = (id_idx+1)%TRAIN_FILE_SIZE;

        /* Conditionally dump this image to HTML */
        if( /* net->label == 6 && */  imageDumpCount < 100 )
        {
          dumpInputLayerToHTML( &net->hyperparams, net->label, net->X, imageDumpCount ? "a" : "w" );
          imageDumpCount++;
        }

        /* Set up the desired net output values for this label */
        for( i=0; i<net->hyperparams.outputLayerCnt; i++ )
          net->Y[i] = (i==net->label) ? 0.99 : 0.01;

        /* Optionally add some noise to the image data prior to learning */
        if( net->hyperparams.noise > 0.0 )
        {
          addNoise( net );
        }
        /* Train the net with the current image */
        forward( net );
        backward( net );

        /* Optionally perform weight decay regularizaton on the net */
        if( net->hyperparams.weightDecay > 0.0 )
          weightDecay( net, net->hyperparams.weightDecay );
      }

      trainGraphY[epoch] = net->outputErrorTotal;
      GRF_quickGraph( TRAIN_GRAPH_PATH, trainGraphX, trainGraphY, epoch, "red", 700, 320, 5000 );

      /**--------------**/
      /**  VALIDATION  **/
      /**--------------**/
      net->state = 1; /* Its validation time! */
      d = 0.0; /* total output error */
      foo = bar = 0.0;
      fooCnt = barCnt = 0;
      bestValidationScore = 0.0;

      /* Loop across images */
      for( imageCount = 0; imageCount < net->hyperparams.validSubsetSize; imageCount++ )
      {
        /* Load an image & label from the data set into the net inputs */
        for( i=0; i<net->hyperparams.inputLayerCnt-1; i++ )
          net->X[i] = imageData->trainImg[id_idx][i];
        net->label = imageData->trainLabel[id_idx];
        id_idx = (id_idx+1)%TRAIN_FILE_SIZE;

        forward( net );
        getDigitClassification( net, &classInfo );

        if( net->label == classInfo.class1  )
        {
          d += 1.0;
          foo += classInfo.conf;
          fooCnt++;
        }
        else
        {
          bar += classInfo.conf;
          barCnt++;
        }
#if( 0 )
        printf( "%sactual label: %d  belief: %d  confidence: %.5lf   2nd choice: %d   ",
                net->label == classInfo.class1 ? "  " : net->label == classInfo.class2 ? ".." : ">>",
                net->label, classInfo.class1, classInfo.conf, classInfo.class2 );
        for( i=0; i<10; i++ )
        {
          printf( "%s%.3lf%s ",
                   i==classInfo.class1 ? "[" : i==classInfo.class2 ? "(" : "",
                   net->Yhat[i],
                   i==classInfo.class1 ? "]" : i==classInfo.class2 ? ")" : "" );
        }
        printf( "\n" );
        fflush( stdout );
#endif
      } /* end for( imageCount = 0; imageCount...  */

      d /= (double)VALID_SUBSET_SIZE;  /* Normalize error term */

      /* Keep track of the highest validation score in the current epoch */
      if( d > bestValidationScore )
        bestValidationScore = d;

#if( 1 )
      printf( "validation: %.5lf%% correct           \n", d*100.0 );
#endif
#if( 1 )
      printf( "avg conf of correct: %.3lf    of wrong: %.3lf\n", foo/(double)fooCnt, bar / (double)barCnt );
#endif
#if( 0 )
      dumpWeightsToSrcFile( net ); /* After one set of training data, store the weights */
#endif
#if( 0 )
      /* Record the network error to our log file for external viewing */
      logToErrorCSVFile( epoch, net->outputErrorTotal, epoch ? "a" : "w" );
#endif

      printf( "\r" );
      fflush( stdout );
      printf( "  epoch %d / %d   E_t: %.8lf      \r", epoch, net->hyperparams.maxEpochs, net->outputErrorTotal );
      fflush( stdout );

#if( 1 )
      validGraphX[epoch] = (double)epoch;
#else
      graphX[epoch] = (clock() - start) / CLOCKS_PER_SEC;
#endif
      validGraphY[epoch] = d;

      epoch++;

      GRF_quickGraph( VALID_GRAPH_PATH, validGraphX, validGraphY, epoch, "red", 700, 320, 5000 );

      /*---------------------------------------------------------------------------------------------------*/
      /* If k-fold cross-validation is enabled, we advance the image data index a user-specified distance, */
      /* so that the next epoch trains on a different set of data. If not enabled, just reset the index to */
      /* the beginning of the data set                                                                     */
      /*---------------------------------------------------------------------------------------------------*/
#if( SKIP_SUBSET_SIZE == 0 )
      id_idx = 0;
#else
      id_idx = (id_idx + net->hyperparams.skipSetSize )%TRAIN_FILE_SIZE;
#endif

      /* EARLY EPOCH STOPPING IF ERROR CREEPS UP AT ALL */
      if( d >= net->hyperparams.stopValidScore )
      {
        epoch = net->hyperparams.maxEpochs; /* bail out */
        printf( "EARLY STOP, because validation score of %.6lf reached\n", net->hyperparams.stopValidScore );
      }
    }  /* end while( epoch < net->hyperparams.maxEpochs )  */

    hpSweep++;

    if( net->hyperparams.maxHpSweeps > 1 )
    {
      /* Log results of hyperparameter sweep */
#if( 0 )
      hpSweepGraphX[hpSweepGraphIdx] = hpSweep-1;
#endif
#if( 1 )
      hpSweepGraphX[hpSweepGraphIdx] = net->hyperparams.randomWtScale;
#endif
      hpSweepGraphY[hpSweepGraphIdx] = bestValidationScore;
      hpSweepGraphIdx++;
      GRF_quickGraph( HPSWEEP_GRAPH_PATH,   hpSweepGraphX,  hpSweepGraphY, hpSweepGraphIdx, "blue", 700, 320, 5000 );

      /* HYPERPARAMETER SWEEP DECISION POINT                               */
      /* Decide if another hyperparameter speep is needed; if not, get out */
      net->hyperparams.randomWtScale  +=    0.005;
      if( net->hyperparams.randomWtScale >  0.3500 )
      {
        hpSweep = net->hyperparams.maxHpSweeps; /* Time to leave */
      }
    }
  } /* end of hyperparameter sweep loop */


  /**----------------------------------**/
  /**  TEST AND SUBMISSION GENERATION  **/
  /**----------------------------------**/
  printf( "\n** test & generate submission **\n" );
  net->state = 2; /* Its test time! */

  testAndGenerateSubmissionFile( net, imageData );


cleanupAndExit:

  /* NETWORK SANITY CHECK */
#if( DEBUG_LEVEL > 0 )
  printf( "\n** Network sanity check **\n" );
  printf( "Input layer bias: %lf\n", net->X[net->hyperparams.inputLayerCnt-1] );
  printf( "Hidden layer bias: %lf\n", net->a2[net->hyperparams.hidden0LayerCnt-1] );
#endif

  if( net )
  {
    free( net );
    net = NULL;
  }
  if( imageData )
  {
    free( imageData );
    imageData = NULL;
  }

  /* Capture the current time, calculate & display how much CPU time we have used */
  end = clock();
  cpuTimeUsed = (end - start) / CLOCKS_PER_SEC;
  cpuDays  = (cpuTimeUsed / (60*60*24));
  cpuHours = (cpuTimeUsed / (60*60))/24;
  cpuMins  = (cpuTimeUsed / 60)%60;
  cpuSecs  = cpuTimeUsed%60;
  printf( "\nCPU time: %.2ldd:%.2ldh:%.2ldm:%.2lds\n\n", cpuDays, cpuHours, cpuMins, cpuSecs );

  return status;
}


/** EOF **/




