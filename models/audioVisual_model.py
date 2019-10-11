import numpy as np
import torch
import os
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from utils.utils import warpgrid
from torch.autograd import Variable

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt

        #initialize model and criterions
        self.net_visual, self.net_unet, self.net_classifier = nets

    def forward(self, input):
        labels = input['labels']
        labels = labels.squeeze(1).long() #covert back to longtensor
        vids = input['vids']
        audio_mags =  input['audio_mags']
        audio_mix_mags = input['audio_mix_mags']
        visuals = input['visuals']
        audio_mix_mags = audio_mix_mags + 1e-10

        # warp the spectrogram
        B = audio_mix_mags.size(0)
        T = audio_mix_mags.size(3)
        if self.opt.log_freq:
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
            audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
            audio_mags = F.grid_sample(audio_mags, grid_warp)

        # calculate ground-truth masks
        gt_masks = audio_mags / audio_mix_mags
        # clamp to avoid large numbers in ratio masks
        gt_masks.clamp_(0., 5.)

        # pass through visual stream and extract visual features
        visual_feature = self.net_visual(Variable(visuals, requires_grad=False))

        # audio-visual feature fusion through UNet and predict mask
        audio_log_mags = torch.log(audio_mix_mags).detach()
        mask_prediction = self.net_unet(audio_log_mags, visual_feature)

        # masking the spectrogram of mixed audio to perform separation
        separated_spectrogram = audio_mix_mags * mask_prediction

        # generate spectrogram for the classifier
        spectrogram2classify = torch.log(separated_spectrogram + 1e-10) #get log spectrogram

        # calculate loss weighting coefficient
        if self.opt.weighted_loss:
            weight = torch.log1p(audio_mix_mags)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = None

        #classify the predicted spectrogram
        label_prediction = self.net_classifier(spectrogram2classify)

        output = {'gt_label': labels, 'pred_label': label_prediction, 'pred_mask': mask_prediction, 'gt_mask': gt_masks, \
                'pred_spectrogram': separated_spectrogram, 'visual_object': visuals, 'audio_mix_mags': audio_mix_mags, 'weight': weight, 'vids': vids}
        return output

