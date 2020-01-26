#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import librosa
import argparse
import numpy as np
import mir_eval.separation

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
	reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
	estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
	(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
	return np.mean(sdr), np.mean(sir), np.mean(sar)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--results_dir', type=str, required=True)
	parser.add_argument('--audio_sampling_rate', type=int, default=11025)
	args = parser.parse_args()

	audio1, _ = librosa.load(os.path.join(args.results_dir, 'audio1_separated.wav'), sr=args.audio_sampling_rate)
	audio2, _ = librosa.load(os.path.join(args.results_dir, 'audio2_separated.wav'), sr=args.audio_sampling_rate)
	audio1_gt, _ = librosa.load(os.path.join(args.results_dir, 'audio1.wav'), sr=args.audio_sampling_rate)
	audio2_gt, _ = librosa.load(os.path.join(args.results_dir, 'audio2.wav'), sr=args.audio_sampling_rate)
	audio_mix, _ = librosa.load(os.path.join(args.results_dir, 'audio_mixed.wav'), sr=args.audio_sampling_rate)

	sdr, sir, sar = getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt)
	sdr_mixed, _, _ = getSeparationMetrics(audio_mix, audio_mix, audio1_gt, audio2_gt)

	output_file = open(os.path.join(args.results_dir, 'eval.txt'),'w')
	output_file.write("%3f %3f %3f %3f %3f" % (sdr, sdr_mixed, sdr - sdr_mixed, sir, sar))
	output_file.close()

if __name__ == '__main__':
	main()
