# from chainer import optimizer, optimizers

from torch import optim as optimizers

import sys

def get_opt(args):
    if args.opt_model=="SGD":
        alpha0 = 0.01   if args.alpha0==0 else args.alpha0
        return optimizers.SGD(lr = alpha0)

    if args.opt_model=="AdaGrad":
        alpha0 = 0.01   if args.alpha0==0 else args.alpha0
        return optimizers.Adagrad(lr = alpha0)

    if args.opt_model=="AdaDelta":
        alpha0 = 0.95   if args.alpha0==0 else args.alpha0
        alpha1 = 1e-06  if args.alpha1==0 else args.alpha1
        return optimizers.Adadelta(rho=alpha0, eps=alpha1)

    if args.opt_model=="Momentum":
        alpha0 = 0.01   if args.alpha0==0 else args.alpha0
        alpha1 = 0.9    if args.alpha1==0 else args.alpha1
        return optimizers.SGD(lr=alpha0, momentum=alpha1)
    
    if args.opt_model=="NAG":
        alpha0 = 0.01   if args.alpha0==0 else args.alpha0
        alpha1 = 0.9    if args.alpha1==0 else args.alpha1
        # return optimizers.NesterovAG(lr=alpha0, momentum=alpha1)
        return optimizers.SGD(lr=alpha0, momentum=alpha1, nesterov=True)

    if args.opt_model=="RMS":
        # return optimizers.RMSpropGraves()
        return optimizers.RMSprop()

    if args.opt_model=="SM":
        # return optimizers.SMORMS3()
        return optimizers.SparseAdam()
        
    if args.opt_model=="Adam": #default case
        alpha0 = 0.001  if args.alpha0==0 else args.alpha0
        alpha1 = 0.9    if args.alpha1==0 else args.alpha1
        alpha2 = 0.999  if args.alpha2==0 else args.alpha2
        alpha3 = 1e-08  if args.alpha3==0 else args.alpha3

        return optimizers.Adam(alpha=alpha0,beta1=alpha1,beta2=alpha2,eps=alpha3)

    print('no such optimization method', args.opt_model)

    sys.exit(1)

def get_adam(Model, alpha0=0.001, alpha1=0.9, alpha2=0.999, alpha3=1e-8):
    opt = optimizers.Adam(alpha=alpha0, beta1=alpha1, beta2=alpha2, eps=alpha3)
    opt.setup(Model)
    return opt
