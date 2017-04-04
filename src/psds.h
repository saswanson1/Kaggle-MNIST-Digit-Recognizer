#ifndef  _PSDS_H_
#define  _PSDS_H_

/* Update Rules */
/* http://sebastianruder.com/optimizing-gradient-descent/  */
/* https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM */
/* https://en.wikipedia.org/wiki/Stochastic_gradient_descent */
/*  Momentum */
/*  Averaging */
/*  Adam: short for Adaptive Moment Estimation */
/*  kSGD: Kalman-based Stochastic Gradient Descent */
/*  RMSProp: Root Mean Square Propagation */
/*  AdaGrad: adaptive gradient algorithm */
/*  Adadelta */
/*  Nesterov accelerated gradient */
/*  Adadelta */

/* UPDATE_RULE_TYPE values */
#define  UR_NOT_DEF  0    /* Not (yet) defined */
#define  UR_ONLINE   1    /* Online; update the weights after each training case */
#define  UR_MBGD     2    /* Mini-Match Gradient Descent; update the weights after a subset of training cases */


/* REGULARIZATION values */
#define  REG_NONE     0   /* Don't use any regularization in the output error calculation */
#define  REG_L1       1   /* Use L1 norm */
#define  REG_L2       2  /* Use L2 norm */

/* FB_IMP definitions */
#define  FBI_DEFAULT  0
#define  FBI_SAS      1
#define  FBI_JB       2
#define  FBI_MM       3
#define  FBI_SCW      4


#endif











