#! -*- coding:utf-8 -*-

from models.ModelA0 import Model as A0, A1

import sys
def get_model(args):
	if args.nn_model=='A0':			
		return A0(args)
	
	if args.nn_model=='A1':			
		return A1(args)

	print('no such model:',args.nn_model)
	sys.exit(1)
