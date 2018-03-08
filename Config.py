class Config(object):
	def __init__(self):
		self.layers = {'conv_out1': 1, #block 1 -- 2 convs and 1 pooling
					   'conv_out2': 0,
					   'conv_out2_pool': 1,
					   'conv_out3': 0, #block 2 -- 2 convs and 1 pooling
					   'conv_out4': 0,
					   'conv_out4_pool': 1,
					   'conv_out5': 0, #block 3 -- 4 convs and 1 pooling
					   'conv_out6': 0,
					   'conv_out7': 0,
					   'conv_out8': 0,
					   'conv_out8_pool': 1,
					   'conv_out9': 0, #block 4 -- 4 convs and 1 pooling
					   'conv_out10': 0,
					   'conv_out11': 0,
					   'conv_out12': 0,
					   'conv_out12_pool': 1,
					   'conv_out13': 0,
					   'conv_out14': 0,
					   'conv_out15': 0,
					   'conv_out16': 0,
					   'conv_out16_pool': 0
		}
		self.img = './imgs/fingerPrint.mat'
		self.recon ='./recons/fingerPrint _recon.npy'
