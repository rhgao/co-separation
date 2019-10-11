import os
import shutil
import librosa
import torch
import cv2
import numpy as np
from torch._six import container_abcs, string_classes, int_classes
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
plt.ioff()

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color

def mkdirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path)

def visualizeSpectrogram(spectrogram, save_path):
	fig,ax = plt.subplots(1,1)
	plt.axis('off')
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	plt.pcolormesh(librosa.amplitude_to_db(spectrogram))
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def istft_reconstruction(mag, phase, hop_length=256, length=65535):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length, length=length)
    return np.clip(wav, -1., 1.)


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
	nets (network list)   -- a list of networks
	requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


#define customized collate to combine useful objects across video pairs
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
def object_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    #print batch
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            return torch.cat([torch.from_numpy(b) for b in batch], 0) #concatenate even if dimension differs
            #return object_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: object_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [object_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))