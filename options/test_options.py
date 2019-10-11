from .base_options import BaseOptions

#test by mix and separate two videos
class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--video1_name', type=str, required=True)
		self.parser.add_argument('--video2_name', type=str) 
		self.parser.add_argument('--output_dir_root', type=str, default='output')
		self.parser.add_argument('--hop_size', default=0.05, type=float, help='the hop length to perform audio separation in a sliding window approach')
		self.parser.add_argument('--subtract_mean', default=True, type=bool, help='subtract channelwise mean from input image')
		self.parser.add_argument('--preserve_ratio', default=False, type=bool, help='whether boudingbox aspect ratio should be preserved when loading')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=False, help='whether to augment input audio/image')
		self.parser.add_argument('--spectrogram_type', type=str, default='magonly', choices=('complex', 'magonly'), help='whether to use magonly or complex spectrogram')
		self.parser.add_argument('--with_discriminator', action='store_true', help='whether to use discriminator')
		self.parser.add_argument('--visualize_spectrogram', action='store_true', help='whether to use discriminator')

		#model specification
		self.parser.add_argument('--visual_pool', type=str, default='maxpool', help='avg or max pool for visual stream feature')
		self.parser.add_argument('--classifier_pool', type=str, default='maxpool', help="avg or max pool for classifier stream feature")
		self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
		self.parser.add_argument('--weights_classifier', type=str, default='', help="weights for audio classifier")
		self.parser.add_argument('--unet_num_layers', type=int, default=7, choices=(5, 7), help="unet number of layers")	
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=1, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")
		self.parser.add_argument('--number_of_classes', default=15, type=int, help='number of classes')
		self.parser.add_argument('--with_silence_category', action='store_true', help='whether to augment input audio/image')	
		self.parser.add_argument('--weighted_loss', action='store_true', help="weighted loss")
		self.parser.add_argument('--binary_mask', action='store_true', help="whether use binary mask, ratio mask is used otherwise")
		self.parser.add_argument('--full_frame', action='store_true', help="pass full frame instead of object regions")
		self.parser.add_argument('--mask_thresh', default=0.5, type=float, help='mask threshold for binary mask')
		self.parser.add_argument('--log_freq', type=bool, default=True, help="whether use log-scale frequency")		
		self.parser.add_argument('--with_frame_feature', action='store_true', help="whether also use frame-level visual feature")	
		self.parser.add_argument('--with_additional_scene_image', action='store_true', help="whether append an extra scene image")
		self.parser.add_argument('--num_of_object_detections_to_use', type=int, default=1, help="num of predictions to avg")
		#include test related hyper parameters here
		self.mode = "test"
