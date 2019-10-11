import os.path
import librosa
from data.base_dataset import BaseDataset
import h5py
import random
from random import randrange
import glob
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch

def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def augment_audio(audio):
    audio = audio * (random.random() + 0.5) # 0.5 - 1.5
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def sample_audio(audio, window):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    return audio_sample

def augment_image(image):
	if(random.random() < 0.5):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
	enhancer = ImageEnhance.Brightness(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	enhancer = ImageEnhance.Color(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	return image

def get_vid_name(npy_path):
    #first 11 chars are the video id
    return os.path.basename(npy_path)[0:11]

def get_clip_name(npy_path):
    return os.path.basename(npy_path)[0:-4]

def get_frame_root(npy_path):
    return os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'frame')

def get_audio_root(npy_path):
    return os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'audio_11025')

def sample_object_detections(detection_bbs):
    class_index_clusters = {} #get the indexes of the detections for each class
    for i in range(detection_bbs.shape[0]):
        if class_index_clusters.has_key(int(detection_bbs[i,1])):
            class_index_clusters[int(detection_bbs[i,1])].append(i)
        else:
            class_index_clusters[int(detection_bbs[i,1])] = [i]
    detection2return = np.array([])
    for cls in class_index_clusters.keys():
        sampledIndex = random.choice(class_index_clusters[cls])
        if detection2return.shape[0] == 0:
            detection2return = np.expand_dims(detection_bbs[sampledIndex,:], axis=0)
        else:
            detection2return = np.concatenate((detection2return, np.expand_dims(detection_bbs[sampledIndex,:], axis=0)), axis=0)
    return detection2return

class AudioVisualMUSICDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.NUM_PER_MIX = opt.num_per_mix
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)

        #initialization
        self.detection_dic = {} #gather the clips for each video
        #load detection hdf5 file
        h5f_path = os.path.join(opt.hdf5_path, opt.mode+'.h5')
        h5f = h5py.File(h5f_path, 'r')
        detections = h5f['detection'][:]
        for detection in detections:
            vidname = get_vid_name(detection) #get video id
            if self.detection_dic.has_key(vidname):
                self.detection_dic[vidname].append(detection)
            else:
                self.detection_dic[vidname] = [detection]

        if opt.mode == 'val':
            vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        elif opt.preserve_ratio:
            vision_transform_list = [transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()]
        else:
            vision_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        if opt.subtract_mean:
            vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.vision_transform = transforms.Compose(vision_transform_list)

        #load hdf5 file of scene images
        if opt.with_additional_scene_image:
            h5f_path = os.path.join(opt.scene_path)
            h5f = h5py.File(h5f_path, 'r')
            self.scene_images = h5f['image'][:]

    def __getitem__(self, index):
        videos2Mix = random.sample(self.detection_dic.keys(), self.NUM_PER_MIX) #get videos to mix
        clip_det_paths = [None for n in range(self.NUM_PER_MIX)]
        clip_det_bbs = [None for n in range(self.NUM_PER_MIX)]
        for n in range(self.NUM_PER_MIX):
            clip_det_paths[n] = random.choice(self.detection_dic[videos2Mix[n]]) #randomly sample a clip
            clip_det_bbs[n] = sample_object_detections(np.load(clip_det_paths[n])) #load the bbs for the clip and sample one from each class

        audios = [None for n in range(self.NUM_PER_MIX)] #audios of mixed videos
        objects_visuals = []
        objects_labels = []
        objects_audio_mag = []
        objects_audio_phase = []
        objects_vids = []
        objects_real_audio_mag = []
        objects_audio_mix_mag = []
        objects_audio_mix_phase = []

        for n in range(self.NUM_PER_MIX):
            vid = random.randint(1,100000000000) #generate a unique video id
            audio_path = os.path.join(get_audio_root(clip_det_paths[n]), get_clip_name(clip_det_paths[n]) + ".wav")
            audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)
            audio_segment= sample_audio(audio, self.audio_window)
            if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
                audio_segment = augment_audio(audio_segment)
            audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)            
            detection_bbs = clip_det_bbs[n]
            audios[n] = audio_segment #make a copy of the audio to mix later
            for i in range(detection_bbs.shape[0]):
                frame_path = os.path.join(get_frame_root(clip_det_paths[n]), get_clip_name(clip_det_paths[n]), str(int(detection_bbs[i,0])).zfill(6) + '.png')
                label = detection_bbs[i,1] - 1 #make the label start from 0
                object_image = Image.open(frame_path).convert('RGB').crop((detection_bbs[i,-4],detection_bbs[i,-3],detection_bbs[i,-2],detection_bbs[i,-1]))
                if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
                    object_image = augment_image(object_image)
                objects_visuals.append(self.vision_transform(object_image).unsqueeze(0))
                objects_labels.append(label)
                #make a copy of the audio spec for each object
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)
            
            #add an additional scene image for each video
            if self.opt.with_additional_scene_image:
                scene_image_path = random.choice(self.scene_images)
                scene_image = Image.open(scene_image_path).convert('RGB')
                if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
                    scene_image = augment_image(scene_image)
                objects_visuals.append(self.vision_transform(scene_image).unsqueeze(0))
                objects_labels.append(self.opt.number_of_classes - 1)
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)

        #mix audio and make a copy of mixed audio spec for each object
        audio_mix = np.asarray(audios).sum(axis=0) / self.NUM_PER_MIX
        audio_mix_mag, audio_mix_phase = generate_spectrogram_magphase(audio_mix, self.stft_frame, self.stft_hop)
        for n in range(self.NUM_PER_MIX):
            detection_bbs = clip_det_bbs[n]
            for i in range(detection_bbs.shape[0]):
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))
            if self.opt.with_additional_scene_image:
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))

        #stack all
        visuals = np.vstack(objects_visuals) #detected objects
        audio_mags = np.vstack(objects_audio_mag) #audio spectrogram magnitude
        audio_phases = np.vstack(objects_audio_phase) #audio spectrogram phase
        labels = np.vstack(objects_labels) #labels for each object, -1 denotes padded object
        vids = np.vstack(objects_vids) #video indexes for each object, each video should have a unique id
        audio_mix_mags = np.vstack(objects_audio_mix_mag)
        audio_mix_phases = np.vstack(objects_audio_mix_phase)

        data = {'labels': labels, 'audio_mags': audio_mags, 'audio_mix_mags': audio_mix_mags, 'vids': vids}
        data['visuals'] = visuals
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            data['audio_phases'] = audio_phases
            data['audio_mix_phases'] = audio_mix_phases
        return data

    def __len__(self):
        if self.opt.mode == 'train':
            return self.opt.batchSize * self.opt.num_batch
        elif self.opt.mode == 'val':
            return self.opt.batchSize * self.opt.validation_batches

    def name(self):
        return 'AudioVisualMUSICDataset'
