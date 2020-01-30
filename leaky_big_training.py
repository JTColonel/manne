from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Input, Dense, Lambda, Concatenate, Dropout, LeakyReLU
from keras.models import Model, Sequential, load_model, clone_model

from keras.regularizers import l2
from keras.losses import mse
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import librosa
import tensorflow as tf

global alpha
global beta
beta = K.variable(3e-7)
alpha = K.variable(0.3)

def change_params(epoch, logs):
	if epoch<=5 and epoch%1==0:
		K.set_value(beta,K.get_value(beta)+2e-5)
	if epoch == 30:
		K.set_value(alpha,0.0)

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename_in', type=str)
	parser.add_argument('--filename_out', type=str)
	parser.add_argument('--net_type', type=str)
	parser.add_argument('--mode', type=str)
	parser.add_argument('--trained_model_name', type=str, default='')
	parser.add_argument('--n_epochs', type=int, default=5)
	parser.add_argument('--skip', type=bool, default=False)
	return parser.parse_args()

class Manne:
	def __init__(self, args):
		self.frames = []
		self.X_train = []
		self.X_val = []
		self.X_test = []
		self.encoder = []
		self.decoder = []
		self.network = []
		self.encoder_widths = []
		self.decoder_widths = []
		
		self.z_mean = K.placeholder(shape=(8,))
		self.z_log_var = K.placeholder(shape=(8,))
		self.beta_changer = []

		self.n_epochs = args.n_epochs
		self.net_type = args.net_type
		self.skip = args.skip
		self.filename_in = args.filename_in
		self.filename_out = args.filename_out
		self.trained_model_name = args.trained_model_name
		
	def do_everything(self):
		self.load_dataset()
		self.define_net()
		self.make_net()
		self.train_net()
		self.evaluate_net()
		self.save_latents()
		
	def just_plot(self):
		self.load_dataset()
		self.load_net()
		self.make_net()
		adam_rate = 5e-4
		self.network.compile(optimizer=Adam(lr=adam_rate), loss=self.my_mse, metrics=[self.my_mse])
		self.evaluate_net()
		self.save_latents()
		
	def sampling(self,args):
		self.z_mean, self.z_log_var = args
		batch = K.shape(self.z_mean)[0]
		dim = K.int_shape(self.z_mean)[1]
		epsilon = K.random_normal(shape=(batch,dim))
		return self.z_mean + K.exp(0.5*self.z_log_var)*epsilon
		
	def get_loss(self, inputs, outputs):
		global beta 
		reconstruction_loss = mse(inputs[:,:2049],outputs)
		kl_loss = 1+self.z_log_var-K.square(self.z_mean)-K.exp(self.z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5*beta
		vae_loss = K.sum(reconstruction_loss+kl_loss)
		return vae_loss
		
	def my_mse(self, inputs, outputs):
		return mse(inputs[:,:2049],outputs)
		
	def my_kl(self, inputs, outputs):
		kl_loss = 1+self.z_log_var-K.square(self.z_mean)-K.exp(self.z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		return kl_loss
	
	def load_net(self):
		enc_filename = os.path.join(os.getcwd(),'models/'+self.trained_model_name+'_trained_encoder.h5')
		print(enc_filename)
		self.encoder = load_model(enc_filename,custom_objects={'sampling': self.sampling}, compile=False)
		dec_filename = os.path.join(os.getcwd(),'models/'+self.trained_model_name+'_trained_decoder.h5')
		self.decoder = load_model(dec_filename,custom_objects={'sampling': self.sampling}, compile=False)
		
	def load_dataset(self):
		filename = 'frames/'+self.filename_in+'_frames.npy'	#Static Data used for training net
		filepath = os.path.join(os.getcwd(),filename)
		orig_frames = np.load(filepath)
		orig_frames = np.asarray(orig_frames)
		len_frames = orig_frames.shape[0]
		
		chroma = np.transpose(librosa.feature.chroma_stft(S=np.transpose(orig_frames), sr=44100))
		chroma = librosa.feature.chroma_stft(S=np.transpose(orig_frames), sr=44100)
		chroma = (chroma == chroma.max(axis=1)[:,None]).astype(int)
		chroma = np.transpose(chroma)	
		augmentations = chroma
	
		self.frames = np.hstack((orig_frames,augmentations))
		
		if args.filename_in == 'one_octave':
			self.X_train = self.frames[:16685,:]
			self.X_val = self.frames[16685:17998,:]
			self.X_test = self.frames[17998:,:]
		elif args.filename_in == 'five_octave':
			self.X_train = self.frames[:78991,:]
			self.X_val = self.frames[78991:84712,:]
			self.X_test = self.frames[84712:,:]
		elif args.filename_in == 'guitar':
			self.X_train = self.frames[:62018,:]
			self.X_val = self.frames[62018:66835,:]
			self.X_test = self.frames[66835:,:]
		elif args.filename_in == 'violin':
			self.X_train = self.frames[:90571,:]
			self.X_val = self.frames[90571:100912,:]
			self.X_test = self.frames[100912:,:]
		else:
			raise Exception('Unexpected filename_in')
			
	def define_net(self):
		if self.net_type=='vae':
			l2_penalty = 0
		else:
			l2_penalty = 1e-7
		
		#8 Neuron Model from the paper
		self.encoder_widths = [1024,512,256,128,64,32,16,8]
		self.decoder_widths = [16,32,64,128,256,512,1024]
		
		#Lighter weight model
		#self.encoder_widths = [512,256,128,64,8]
		#self.decoder_widths = [64,128,256,512]
		
		decoder_outdim = 2049
		drop = 0.0
		alpha_val=0.1
		
		input_spec = Input(shape=(self.frames.shape[1],))
		encoded = Dense(units=self.encoder_widths[0], 
				activation=None,
				kernel_regularizer=l2(l2_penalty))(input_spec)
		encoded = LeakyReLU(alpha=alpha_val)(encoded)
		for width in self.encoder_widths[1:-1]:
			encoded = Dense(units=width, 
				activation=None,
				kernel_regularizer=l2(l2_penalty))(encoded)
			encoded = LeakyReLU(alpha=alpha_val)(encoded)
			
		encoded = Dense(units=self.encoder_widths[-1], activation='sigmoid', kernel_regularizer=l2(l2_penalty))(encoded)
		
		if self.net_type == 'vae':
			self.z_mean = Dense(self.encoder_widths[-1],input_shape=(self.encoder_widths[-1],), name='z_mean')(encoded)
			self.z_log_var = Dense(self.encoder_widths[-1],input_shape=(self.encoder_widths[-1],), name='z_log_var')(encoded)
			z = Lambda(self.sampling,output_shape=(self.encoder_widths[-1],), name='z')([self.z_mean,self.z_log_var])
			self.encoder = Model(input_spec, [self.z_mean, self.z_log_var, z])	
		else:
			self.encoder = Model(input_spec, encoded)
		
		if self.skip == True:
			input_latent = Input(shape=(self.encoder_widths[-1]+12,))
		else:
			input_latent = Input(shape=(self.encoder_widths[-1],))
		
		decoded = Dense(units=self.decoder_widths[0], 
			activation=None,
			kernel_regularizer=l2(l2_penalty))(input_latent)
		decoded = LeakyReLU(alpha=alpha_val)(decoded)
		for width in self.decoder_widths[1:]:
			decoded = Dense(units=width, 
				activation=None,
				kernel_regularizer=l2(l2_penalty))(decoded)
			decoded = LeakyReLU(alpha=alpha_val)(decoded)
		decoded = Dense(units=2049, 
			activation='relu',
			kernel_regularizer=l2(l2_penalty))(decoded)
		self.decoder = Model(input_latent,decoded)
		
	def make_net(self):
		auto_input = Input(shape=(self.frames.shape[1],))
		encoded = self.encoder(auto_input)
		
		if self.net_type == 'vae':
			latents = encoded[2]
		else:
			latents = encoded
			
		if self.skip == True:
			chroma_input = Input(shape=(12,))
			new_latents = Concatenate()([latents,chroma_input])
			decoded = self.decoder(new_latents)
			self.network = Model(inputs=[auto_input,chroma_input], outputs=decoded)
		else:
			decoded = self.decoder(latents)
			self.network = Model(inputs=[auto_input], outputs=decoded)

		print('\n net summary \n')
		self.network.summary()
		print('\n encoder summary \n')
		self.encoder.summary()
		print('\n decoder summary \n')
		self.decoder.summary()
		
	def train_net(self):
		adam_rate = 5e-4
		if self.skip == True: #Handling case where Keras expects two inputs
			train_data = [self.X_train,self.X_train[:,-12:]]
			val_data = [self.X_val,self.X_val[:,-12:]]
		else:
			train_data = self.X_train
			val_data = self.X_val
		if self.net_type == 'vae':
			beta_changer = LambdaCallback(on_epoch_end=change_params)
			self.network.compile(optimizer=Adam(lr=adam_rate), loss=self.get_loss, metrics=[self.my_mse, self.my_kl])
			self.network.fit(x=train_data, y=self.X_train,
					epochs=self.n_epochs,
					batch_size=200,
					shuffle=True,
					validation_data=(val_data, self.X_val),
					callbacks=[beta_changer]
					)
			
		else:
			alpha_changer = LambdaCallback(on_epoch_end=change_params)
			self.network.compile(optimizer=Adam(lr=adam_rate), loss=self.my_mse, metrics=[self.my_mse])
			self.network.fit(x=train_data, y=self.X_train,
					epochs=self.n_epochs,
					batch_size=200,
					shuffle=True,
					validation_data=(val_data, self.X_val),
					callbacks=[alpha_changer]
					)		
		self.encoder.save('models/'+self.net_type+'_'+self.filename_out+'_trained_encoder.h5')
		self.decoder.save('models/'+self.net_type+'_'+self.filename_out+'_trained_decoder.h5')
		
	def save_latents(self):

		indat = self.frames
		enc_mag = self.encoder.predict(indat,verbose=1)
		
		if self.net_type == 'vae':
			a = enc_mag[0]
			b = enc_mag[1]
			print(a.shape)
			print(b.shape)
			enc_mag = np.hstack((enc_mag[0],enc_mag[1]))
			
		df = pd.DataFrame(enc_mag)
		df.to_csv('encoded_mags.csv')
			

	def evaluate_net(self):
		if self.skip == True: #Handling case where Keras expects two inputs
			test_data = [self.X_test,self.X_test[:,-12:]]
			val_data = [self.X_val,self.X_val[:,-12:]]
		else:
			test_data = self.X_test
			val_data = self.X_val
			
		if args.filename_in == 'one_octave':
			mod = 1
		elif args.filename_in == 'five_octave' or args.filename_in == 'violin':
			mod = 10
		elif args.filename_in == 'guitar':
			mod = 3
		else:
			mod = 1
		
		print('\n')
		print('Evaluating performance on validation and test sets')
		a=self.network.evaluate(x=val_data,y=self.X_val,verbose=1)
		b=self.network.evaluate(x=test_data,y=self.X_test,verbose=1)
		print('\n')
		for idx in range(len(self.network.metrics_names)):
			print('Validation '+self.network.metrics_names[idx])
			print(a[idx])
		print('\n')
		for idx in range(len(self.network.metrics_names)):
			print('Testing '+self.network.metrics_names[idx])
			print(b[idx])
		print('\n')
		print('Plotting network reconstructions')
		valset_eval = self.network.predict(val_data,verbose=1)
		testset_eval = self.network.predict(test_data,verbose=1)
		frame_check = [100, 150, 200, 250, 300, 350, 400, 450, 500]

		for frame in frame_check:
			frame *= mod
			xx = np.arange(2049)*(22050/2049)
			val_yy = self.X_val[frame,0:2049]
			val_zz = valset_eval[frame,0:2049]
			test_yy = self.X_val[frame,0:2049]
			test_zz = valset_eval[frame,0:2049]
			plt.figure(1)
			plt.subplot(211)
			plt.plot(xx,val_yy)
			plt.ylim([0,1.2])
			plt.ylabel('Spectral Magnitude')
			plt.xscale('log')
			plt.xlabel('Frequency (Hz)')
			plt.title('Input Spectrum')
			plt.subplot(212)
			plt.plot(xx,val_zz,color='r')
			plt.ylim([0,1.2])
			plt.ylabel('Spectral Magnitude')
			plt.xscale('log')
			plt.xlabel('Frequency (Hz)')
			plt.title('Output Spectrum')
			plt.tight_layout()
			plotname = self.net_type+'_val_'+str(frame)+'.pdf'
			plt.savefig(plotname, format = 'pdf', bbox_inches='tight')
			plt.clf()
			
			plt.figure(1)
			plt.subplot(211)
			plt.plot(xx,test_yy)
			plt.ylim([0,1.2])
			plt.ylabel('Spectral Magnitude')
			plt.xscale('log')
			plt.xlabel('Frequency (Hz)')
			plt.title('Input Spectrum')
			plt.subplot(212)
			plt.plot(xx,test_zz,color='r')
			plt.ylim([0,1.2])
			plt.ylabel('Spectral Magnitude')
			plt.xscale('log')
			plt.xlabel('Frequency (Hz)')
			plt.title('Output Spectrum')
			plt.tight_layout()
			plotname = self.net_type+'_test_'+str(frame)+'.pdf'
			plt.savefig(plotname, format = 'pdf', bbox_inches='tight')
			plt.clf()
			

if __name__ == '__main__':
	args = get_arguments()
	my_manne = Manne(args)
	if args.mode == 'train':
		my_manne.do_everything()
	else:
		my_manne.just_plot()






