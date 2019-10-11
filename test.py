#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import librosa
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from options.test_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_magphase
from utils import utils

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def get_separated_audio(outputs, batch_data, opt):
	# fetch data and predictions
	mag_mix = batch_data['audio_mix_mags']
	phase_mix = batch_data['audio_mix_phases']
	pred_masks_ = outputs['pred_mask']
	mag_mix_ = outputs['audio_mix_mags']
	# unwarp log scale
	B = mag_mix.size(0)
	if opt.log_freq:
		grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, pred_masks_.size(3), warp=False)).to(opt.device)
		pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
	else:
		pred_masks_linear = pred_masks_
	# convert into numpy
	mag_mix = mag_mix.numpy()
	phase_mix = phase_mix.numpy()
	pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
	pred_mag = mag_mix[0, 0] * pred_masks_linear[0, 0]
	preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=opt.stft_hop, length=opt.audio_window)
	return preds_wav

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
        reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
        #print reference_sources.shape
        estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
        #print estimated_sources.shape
        (sdr, sir, sar, perm) = bss_eval_sources(np.asarray(reference_sources), np.asarray(estimated_sources), False)
        #print sdr, sir, sar, perm
        return np.mean(sdr), np.mean(sir), np.mean(sar)

def main():
	#load test arguments
	opt = TestOptions().parse()
	opt.device = torch.device("cuda")

	# Network Builders
	builder = ModelBuilder()
	net_visual = builder.build_visual(
	        pool_type=opt.visual_pool,
	        weights=opt.weights_visual)
	net_unet = builder.build_unet(
	        unet_num_layers = opt.unet_num_layers,
	        ngf=opt.unet_ngf,
	        input_nc=opt.unet_input_nc,
	        output_nc=opt.unet_output_nc,
	        weights=opt.weights_unet)
	if opt.with_additional_scene_image:
	    opt.number_of_classes = opt.number_of_classes + 1
	net_classifier = builder.build_classifier(
	        pool_type=opt.classifier_pool,
	        num_of_classes=opt.number_of_classes,
	        input_channel=opt.unet_output_nc,
	        weights=opt.weights_classifier)
	nets = (net_visual, net_unet, net_classifier)

	# construct our audio-visual model
	model = AudioVisualModel(nets, opt)
	model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
	model.to(opt.device)
	model.eval()

	#load the two audios
	audio1_path = os.path.join(opt.data_path, 'audio_11025', opt.video1_name + '.wav')
	audio1, _ = librosa.load(audio1_path, sr=opt.audio_sampling_rate)
	audio2_path = os.path.join(opt.data_path, 'audio_11025', opt.video2_name + '.wav')
	audio2, _ = librosa.load(audio2_path, sr=opt.audio_sampling_rate)

	#make sure the two audios are of the same length and then mix them
	audio_length = min(len(audio1), len(audio2))
	audio1 = clip_audio(audio1[:audio_length])
	audio2 = clip_audio(audio2[:audio_length])
	audio_mix = (audio1 + audio2) / 2.0

	#define the transformation to perform on visual frames
	vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
	if opt.subtract_mean:
		vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	vision_transform = transforms.Compose(vision_transform_list)

	#load the object regions of the highest confidence score for both videos
	detectionResult1 = np.load(os.path.join(opt.data_path, 'detection_results', opt.video1_name + '.npy'))
	detectionResult2 = np.load(os.path.join(opt.data_path, 'detection_results', opt.video2_name + '.npy'))

	avged_sep_audio1 = np.zeros((audio_length))
	avged_sep_audio2 = np.zeros((audio_length))

	for i in range(opt.num_of_object_detections_to_use):
		det_box1 = detectionResult1[np.argmax(detectionResult1[:,2]),:] #get the box of the highest confidence score
		det_box2 = detectionResult2[np.argmax(detectionResult2[:,2]),:] #get the box of the highest confidence score
		detectionResult1[np.argmax(detectionResult1[:,2]),2] = 0 # set to 0 after using it
		detectionResult2[np.argmax(detectionResult2[:,2]),2] = 0 # set to 0 after using it
		frame_path1 = os.path.join(opt.data_path, 'frame', opt.video1_name, "%06d.png" % det_box1[0])
		frame_path2 = os.path.join(opt.data_path, 'frame', opt.video2_name, "%06d.png" % det_box2[0])
		detection1 = Image.open(frame_path1).convert('RGB').crop((det_box1[-4],det_box1[-3],det_box1[-2],det_box1[-1]))
		detection2 = Image.open(frame_path2).convert('RGB').crop((det_box2[-4],det_box2[-3],det_box2[-2],det_box2[-1]))

		#perform separation over the whole audio using a sliding window approach
		overlap_count = np.zeros((audio_length))
		sep_audio1 = np.zeros((audio_length))
		sep_audio2 = np.zeros((audio_length))
		sliding_window_start = 0
		data = {}
		samples_per_window = opt.audio_window
		while sliding_window_start + samples_per_window < audio_length:
			sliding_window_end = sliding_window_start + samples_per_window
			audio_segment = audio_mix[sliding_window_start:sliding_window_end]
			audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
			data['audio_mix_mags'] = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
			data['audio_mix_phases'] = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
			data['real_audio_mags'] = data['audio_mix_mags'] #dont' care for testing
			data['audio_mags'] = data['audio_mix_mags'] #dont' care for testing
			#separate for video 1
			data['visuals'] = vision_transform(detection1).unsqueeze(0)
			data['labels'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
			data['vids'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
			outputs = model.forward(data)
			reconstructed_signal = get_separated_audio(outputs, data, opt)
			sep_audio1[sliding_window_start:sliding_window_end] = sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal
			#separate for video 2
			data['visuals'] = vision_transform(detection2).unsqueeze(0)
			#data['label'] = torch.LongTensor([0]) #don't care for testing
			outputs = model.forward(data)
			reconstructed_signal = get_separated_audio(outputs, data, opt)
			sep_audio2[sliding_window_start:sliding_window_end] = sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal
			#update overlap count
			overlap_count[sliding_window_start:sliding_window_end] = overlap_count[sliding_window_start:sliding_window_end] + 1
			sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

		#deal with the last segment
		audio_segment = audio_mix[-samples_per_window:]
		audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
		data['audio_mix_mags'] = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
		data['audio_mix_phases'] = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
		data['real_audio_mags'] = data['audio_mix_mags'] #dont' care for testing
		data['audio_mags'] = data['audio_mix_mags'] #dont' care for testing
		#separate for video 1
		data['visuals'] = vision_transform(detection1).unsqueeze(0)
		data['labels'] = torch.FloatTensor(np.ones((1,1))) #don't care for testing
		data['vids'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
		outputs = model.forward(data)
		reconstructed_signal = get_separated_audio(outputs, data, opt)
		sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal
		#separate for video 2
		data['visuals'] = vision_transform(detection2).unsqueeze(0)
		outputs = model.forward(data)
		reconstructed_signal = get_separated_audio(outputs, data, opt)
		sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal
		#update overlap count
		overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

		#divide the aggregated predicted audio by the overlap count
		avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count) * 2)
		avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count) * 2)


	separation1 = avged_sep_audio1 / opt.num_of_object_detections_to_use
	separation2 = avged_sep_audio2 / opt.num_of_object_detections_to_use

	#output original and separated audios
	output_dir = os.path.join(opt.output_dir_root, opt.video1_name + 'VS' + opt.video2_name)
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	librosa.output.write_wav(os.path.join(output_dir, 'audio1.wav'), audio1, opt.audio_sampling_rate)
	librosa.output.write_wav(os.path.join(output_dir, 'audio2.wav'), audio2, opt.audio_sampling_rate)
	librosa.output.write_wav(os.path.join(output_dir, 'audio_mixed.wav'), audio_mix, opt.audio_sampling_rate)
	librosa.output.write_wav(os.path.join(output_dir, 'audio1_separated.wav'), separation1, opt.audio_sampling_rate)
	librosa.output.write_wav(os.path.join(output_dir, 'audio2_separated.wav'), separation2, opt.audio_sampling_rate)
	#save the two detections
	detection1.save(os.path.join(output_dir,  'audio1.png'))
	detection2.save(os.path.join(output_dir, 'audio2.png'))
	#save the spectrograms & masks
	if opt.visualize_spectrogram:
		import matplotlib.pyplot as plt
		plt.switch_backend('agg')
		plt.ioff()
		audio1_mag = generate_spectrogram_magphase(audio1, opt.stft_frame, opt.stft_hop, with_phase=False)
		audio2_mag = generate_spectrogram_magphase(audio2, opt.stft_frame, opt.stft_hop, with_phase=False)
		audio_mix_mag = generate_spectrogram_magphase(audio_mix, opt.stft_frame, opt.stft_hop, with_phase=False)
		separation1_mag = generate_spectrogram_magphase(separation1, opt.stft_frame, opt.stft_hop, with_phase=False)
		separation2_mag = generate_spectrogram_magphase(separation2, opt.stft_frame, opt.stft_hop, with_phase=False)
		utils.visualizeSpectrogram(audio1_mag[0,:,:], os.path.join(output_dir, 'audio1_spec.png'))
		utils.visualizeSpectrogram(audio2_mag[0,:,:], os.path.join(output_dir, 'audio2_spec.png'))
		utils.visualizeSpectrogram(audio_mix_mag[0,:,:], os.path.join(output_dir, 'audio_mixed_spec.png'))
		utils.visualizeSpectrogram(separation1_mag[0,:,:], os.path.join(output_dir, 'separation1_spec.png'))
		utils.visualizeSpectrogram(separation2_mag[0,:,:], os.path.join(output_dir, 'separation2_spec.png'))

if __name__ == '__main__':
    main()
