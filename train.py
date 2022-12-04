import argparse
import functools
import gzip
import http.client
import io
import math
import operator
import os
import signal
import tarfile
import urllib.request
import warnings

import torch

# Constants ####################################################################

TIMEOUT = 5
MNIST_BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
MNIST_IMAGES_MAGIC = 0x803
MNIST_LABELS_MAGIC = 0x801
MNIST_N_ROWS = 28
MNIST_N_COLS = 28
MNIST_N_CLASSES = 10
MNIST_N_TRAIN = 60_000
MNIST_N_TEST = 10_000
MNIST_N_CHANNELS = 1
CIFAR_BASE_URL = 'https://www.cs.toronto.edu/~kriz/'
CIFAR_N_ROWS = 32
CIFAR_N_COLS = 32
CIFAR_N_CLASSES = 10
CIFAR_N_TRAIN = 50_000
CIFAR_N_TEST = 10_000
CIFAR_N_CHANNELS = 3

# Utility ######################################################################

def signal_handler(sig, frame):
	print()
	raise SystemExit(0)

def debug_signal_handler(sig, frame):
	breakpoint()

def download_or_read(base_url: str, file_name: str) -> io.BufferedIOBase:
	if os.path.exists(file_name):
		return open(file_name, 'rb')
	else:
		with urllib.request.urlopen(base_url + file_name, b'', TIMEOUT) as resp, \
			open(file_name, 'wb') as f:
			resp: http.client.HTTPResponse
			buf = resp.read()
			f.write(buf)
			return io.BytesIO(buf)

def write_ppm(file_name: str, values, width: int, height: int) -> None:
	'''Takes a flattened image and creates a ppm file to visualize it.'''
	with open(file_name, 'wb') as f:
		f.write(f'P6\n# comment\n{width} {height} {255}\n'.encode())
		for value in values: f.write(bytes([value, value, value]))

def traverse_graph(grad_fn, depth=0) -> None:
	# TODO: this should output graphviz data.
	if grad_fn is None: return
	print(f'{" "*depth}{grad_fn}')

	for other_grad_fn, _ in grad_fn.next_functions:
		traverse_graph(other_grad_fn, depth+1)

def count_parameters(net: torch.nn.Module) -> int:
	res = sum(functools.reduce(operator.mul, p.shape) for p in net.parameters())
	return res

class Debug(torch.nn.Module):
	def __init__(self, should_break: bool) -> None:
		super().__init__()
		self.should_break = should_break

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		print(X.size())
		if self.should_break:
			breakpoint()
		return X

class ArgMax(torch.nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		res = X.argmax(dim=self.dim)
		return res

class OneHot(torch.nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.num_classes = num_classes

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		res = torch.nn.functional.one_hot(X, self.num_classes).float()
		return res

def read_mnist_images(buf: memoryview) -> memoryview:
	# TODO: return meaningful errors.
	if len(buf) <= 16:
		raise Exception('Bad MNIST image file.')

	magic_number = int.from_bytes(buf[0:4], 'big', signed=False)
	n_imgs = int.from_bytes(buf[4:8], 'big', signed=False)
	n_rows = int.from_bytes(buf[8:12], 'big', signed=False)
	n_cols = int.from_bytes(buf[12:16], 'big', signed=False)
	n_pixels = n_rows*n_cols
	images_data = buf[16:]

	if magic_number != MNIST_IMAGES_MAGIC:
		raise Exception('Bad MNIST image file.')
	if n_rows != MNIST_N_ROWS:
		raise Exception('Bad MNIST image file.')
	if n_cols != MNIST_N_COLS:
		raise Exception('Bad MNIST image file.')
	if n_imgs != len(images_data)//n_pixels:
		raise Exception('Bad MNIST image file.')

	return images_data

def read_mnist_labels(buf: memoryview) -> memoryview:
	# TODO: return meaningful errors.
	if len(buf) <= 8:
		raise Exception('Bad MNIST label file.')

	magic_number = int.from_bytes(buf[0:4], 'big', signed=False)
	n_items = int.from_bytes(buf[4:8], 'big', signed=False)
	labels_data = buf[8:]

	if magic_number != MNIST_LABELS_MAGIC:
		raise Exception('Bad MNIST label file.')
	if n_items != len(labels_data):
		raise Exception('Bad MNIST label file.')
	if any(byte >= MNIST_N_CLASSES for byte in labels_data):
		raise Exception('Bad MNIST label file.')

	return labels_data

# Models #######################################################################

def make_sparse_proj_matrix(nrows: int, ncols: int, s: int) -> torch.Tensor:
	# ugly ugly function
	import random
	import math
	random.seed(42)

	col_indices = []
	row_indices = []
	for i in range(nrows*ncols):
		if random.uniform(0, 1) < 1/s:
			col_indices.append(i%ncols)
			row_indices.append(i//ncols)

	values = [-math.sqrt(s) if random.uniform(0, 1) < .5 else math.sqrt(s)
		for _ in col_indices]

	size = (nrows, ncols)
	res = torch.sparse_coo_tensor([row_indices, col_indices], values, size) \
		.to_sparse_csr()
	return res

@torch.no_grad()
def init_projection(param: torch.Tensor) -> None:
	if __debug__: initial_shape = param.shape
	torch.nn.init.uniform_(param)
	# Normalizing columns to unit length.
	cols_norm = torch.norm(param, dim=1)
	param.transpose_(1, 0)
	param.div_(cols_norm)
	param.transpose_(1, 0)
	assert param.shape == initial_shape

def project(
		initial: torch.Tensor,
		project: torch.Tensor,
		current: torch.Tensor
	) -> torch.Tensor:
	'''Implementation of intrinsic dimension projection as found in the authors'
	code (commit: 2a7ebd257921e3ad7098316d930ae984c9426d49):
	  * https://github.com/uber-research/intrinsic-dimension/blob/master/keras_ext/rproj_layers.py#L102
	  * https://github.com/uber-research/intrinsic-dimension/blob/master/keras_ext/rproj_layers_util.py#L108
	  * https://github.com/uber-research/intrinsic-dimension/blob/master/keras_ext/rproj_layers.py#L182
	  * https://github.com/uber-research/intrinsic-dimension/blob/master/intrinsic_dim/model_builders.py#L160
	This function projects a parameter (or weight/bias) from the dimension of
	theta_d (the instrinsic one) to the one of theta_D_0 (the dimension used by
	the network).'''
	# NOTE: instead of adding a weird dimension here I could use torch.unsqueeze(weight, 0) here.
	assert current.requires_grad
	assert not project.requires_grad
	assert not initial.requires_grad
	offset = (current @ project).view(initial.shape)
	res = initial + offset
	return res

class IntrLinear(torch.nn.Module):
	def __init__(
		self,
		current: torch.nn.Parameter,
		in_features: int,
		out_features: int,
		bias: bool = True,
		device = None,
		dtype = None
	) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super().__init__()
		assert current.requires_grad
		intr_dim = current.size()[1]
		self.register_parameter('current', current)
		self.register_buffer('initial_weight',
			torch.empty((out_features, in_features), **factory_kwargs))
		self.register_buffer('project_weight',
			torch.empty((intr_dim, in_features*out_features), **factory_kwargs))
		if bias:
			self.bias = True
			self.register_buffer('initial_bias',
				torch.empty(out_features, **factory_kwargs))
			self.register_buffer('project_bias',
				torch.empty((intr_dim, out_features), **factory_kwargs))
		else:
			self.bias = False
		self.reset_parameters()

	def reset_parameters(self) -> None:
		init_projection(self.project_weight)
		torch.nn.init.kaiming_uniform_(self.initial_weight, a=math.sqrt(5))
		if self.bias is not None:
			init_projection(self.project_bias)
			# NOTE: this is a bit different from how torch.nn.Linear does it.
			torch.nn.init.uniform_(self.initial_bias)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		projected_weight = project(self.initial_weight, self.project_weight,
			self.current)
		if self.bias is not None:
			projected_bias = project(self.initial_bias, self.project_bias,
				self.current)
		else:
			projected_bias = None
		return torch.nn.functional.linear(input, projected_weight, projected_bias)

class IntrConv2d(torch.nn.Module):
	def __init__(
			self,
			current: torch.nn.Parameter,
			in_channels: int,
			out_channels: int,
			kernel_size: tuple[int, ...],
			stride: tuple[int, ...] = (1, 1),
			padding: tuple[int, ...] = (0, 0),
			dilation: tuple[int, ...] = (1, 1),
			transposed: bool = False,
			output_padding: tuple[int, ...] = (0, 0),
			groups: int = 1,
			bias: bool = True,
			padding_mode: str = 'zeros',
			device=None,
			dtype=None) -> None:
		super().__init__()
		factory_kwargs = {'device': device, 'dtype': dtype}
		# Code copied from torch.nn.modules.conv._ConvNd
		if groups <= 0:
			raise ValueError('groups must be a positive integer')
		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		valid_padding_strings = {'same', 'valid'}
		if isinstance(padding, str):
			if padding not in valid_padding_strings:
				raise ValueError(
					"Invalid padding string {!r}, should be one of {}".format(
						padding, valid_padding_strings))
			if padding == 'same' and any(s != 1 for s in stride):
				raise ValueError("padding='same' is not supported for strided convolutions")
		valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
		if padding_mode not in valid_padding_modes:
			raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
				valid_padding_modes, padding_mode))

		if isinstance(kernel_size, int):
			kernel_size = (kernel_size, kernel_size)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.transposed = transposed
		self.output_padding = output_padding
		self.groups = groups
		self.padding_mode = padding_mode
		# `_reversed_padding_repeated_twice` is the padding to be passed to
		# `F.pad` if needed (e.g., for non-zero padding types that are
		# implemented as two ops: padding + conv). `F.pad` accepts paddings in
		# reverse order than the dimension.
		if isinstance(self.padding, str):
			self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
			if padding == 'same':
				for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
					total_padding = d * (k - 1)
					left_pad = total_padding // 2
					self._reversed_padding_repeated_twice[2 * i] = left_pad
					self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
		else:
			self._reversed_padding_repeated_twice = type(self)._reverse_repeat_tuple(self.padding, 2)

		# My code starts here.
		assert len(kernel_size) == 2
		a, b = (in_channels, out_channels // groups) if transposed \
			else (out_channels, in_channels // groups)
		intr_dim = current.size()[1]
		self.register_buffer('initial_weight',
			torch.empty(a, b, *kernel_size, **factory_kwargs))
		self.register_buffer('project_weight',
			torch.empty((intr_dim, a * b * kernel_size[0] * kernel_size[1])))
		self.register_parameter('current', current)
		if bias:
			self.bias = True
			self.register_buffer('initial_bias',
				torch.empty(out_channels, **factory_kwargs))
			self.register_buffer('project_bias',
				torch.empty((intr_dim, out_channels), **factory_kwargs))
		else:
			self.bias = False

		self.reset_parameters()

	def reset_parameters(self):
		init_projection(self.project_weight)
		torch.nn.init.kaiming_uniform_(self.initial_weight, a=math.sqrt(5))
		if self.bias is not None:
			init_projection(self.project_bias)
			# NOTE: this is a bit different from how torch.nn.modules.conv._ConvNd does it.
			torch.nn.init.uniform_(self.initial_bias)

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		assert self.padding_mode == 'zeros'
		assert X.dim() == 4
		projected_weight = project(self.initial_weight, self.project_weight, self.current)
		if self.bias is not None:
			projected_bias = project(self.initial_bias, self.project_bias, self.current)
		else:
			projected_bias = None
		return torch.nn.functional.conv2d(X, projected_weight, projected_bias,
			self.stride, self.padding, self.dilation, self.groups)

def make_fcnet(
		intr_dim: int,
		n_inputs: int,
		width: int,
		depth: int,
		n_classes: int
	) -> torch.nn.Module:
	if intr_dim > 0:
		current = torch.nn.Parameter(torch.zeros((1, intr_dim), requires_grad=True))
		Linear = functools.partial(IntrLinear, current)
	else:
		Linear = torch.nn.Linear

	args = [
		torch.nn.Flatten(1, -1),
		Linear(n_inputs, width),
		torch.nn.ReLU(),
	]
	for _ in range(depth):
		args.append(Linear(width, width))
		args.append(torch.nn.ReLU())
	args.append(Linear(width, n_classes)) # to output logits

	res = torch.nn.Sequential(*args)
	return res

def make_lenet(
		intr_dim: int,
		n_channels: int,
		n_classes: int,
		n_rows: int,
		n_cols: int
	) -> torch.nn.Module:
	current = torch.nn.Parameter(torch.zeros((1, intr_dim), requires_grad=True))
	Conv2d = functools.partial(IntrConv2d, current) if intr_dim else torch.nn.Conv2d
	Linear = functools.partial(IntrLinear, current) if intr_dim else torch.nn.Linear
	# After all Conv2d and MaxPool2d
	rows = n_rows - 12
	cols = n_cols - 12
	res = torch.nn.Sequential(
		Conv2d(in_channels=n_channels, out_channels=6, kernel_size=5, stride=1, padding='valid'),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d(kernel_size=2),
		Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding='valid'),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d(kernel_size=2),
		torch.nn.Flatten(),
		Linear(rows*cols, 120),
		torch.nn.ReLU(),
		Linear(120, 84),
		torch.nn.ReLU(),
		Linear(84, n_classes) # logits
	)
	return res

def make_small_lenet(
		n_channels: int,
		n_classes: int,
		n_rows: int,
		n_cols: int
	) -> torch.nn.Module:
	# Ugly hackity hack!
	if n_channels == 1:
		# After all Conv2d and MaxPool2d
		rows = n_rows - 24
		cols = n_cols - 24
		res = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=5, stride=1, padding='valid'),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2),
			torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding='valid'),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2),
			torch.nn.Flatten(),
			torch.nn.Linear(rows*cols, n_classes) # logits
		)
		return res
	elif n_channels == 3:
		res = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=5, stride=1, padding='valid'),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2),
			torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding='valid'),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2),
			torch.nn.Flatten(), # batchsize * flattened_dim
			torch.nn.Linear(75, 20),
			torch.nn.ReLU(),
			torch.nn.Linear(20, n_classes) # logits
		)
		return res
	else: assert False

def train(
		net: torch.nn.Module,
		epochs: int,
		patience: int,
		train_dataloader: torch.utils.data.DataLoader,
		logit_normalizer: torch.nn.Module,
		label_postproc: torch.nn.Module,
		criterion: torch.nn.Module,
		optimizer, # no type here :( torch.optimizer.Optimizer
		test_dataloader: torch.utils.data.DataLoader
	) -> float:
	patience_kept = 0
	best_epoch = 0
	best_accuracy = float('-inf')

	for epoch in range(epochs):
		if patience_kept >= patience: break

		net.train()
		losses: list[float] = []
		for i, (data, labels) in enumerate(train_dataloader):
			# NOTE: should tensors be moved here to device instead of a priori?
			data: torch.Tensor
			labels: torch.Tensor

			logits = net(data)
			predictions = logit_normalizer(logits)

			loss = criterion(predictions, label_postproc(labels))
			losses.append(loss.item())

			loss.backward()

			optimizer.step()
			optimizer.zero_grad()
		avg_loss = torch.mean(torch.tensor(losses))

		net.eval()
		correct = 0
		with torch.no_grad():
			for data, labels in test_dataloader:
				predictions = net(data).argmax(dim=1)

				# assert (predictions < N_CLASSES).all()
				# assert (labels < N_CLASSES).all()
				assert predictions.shape == labels.shape

				correct += (predictions == labels).sum().item()
		accuracy = correct/len(test_dataloader.dataset)
		assert accuracy <= 1.0

		if accuracy > best_accuracy:
			patience_kept = 0
			# best_params = net.params()
			best_epoch = epoch
			best_accuracy = accuracy
			marker = ' *'
		else:
			patience_kept += 1
			marker = ''

		print(f'{epoch=:02} {accuracy=:.3f} {avg_loss=:.3f}{marker}')

	print(f'{best_epoch=} {best_accuracy=:.3f}')
	return best_accuracy

# Main #########################################################################

def main() -> int:
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGUSR1, debug_signal_handler)

	parser = argparse.ArgumentParser()
	parser.add_argument('-data', action='store', choices=['mnist', 'cifar'], required=True)
	parser.add_argument('-model', action='store', choices=['fc', 'lenet'], required=True)
	parser.add_argument('-epochs', action='store', type=int, required=True)
	parser.add_argument('-patience', action='store', type=int, required=True)
	parser.add_argument('-lr', action='store', type=float, required=True)
	parser.add_argument('-intr', action='store', type=int, default=0)
	parser.add_argument('-batch', action='store', type=int, default=128)
	parser.add_argument('-device', action='store', choices=['cpu', 'cuda', 'mps'], default='cpu')
	parser.add_argument('-small', action='store_true')
	parser.add_argument('-random', action='store_true')
	parser.add_argument('-dry', action='store_true')
	parser.add_argument('-peek', action='store_true')
	parser.add_argument('-load', action='store_true')

	args = parser.parse_args()

	seed = int.from_bytes(os.urandom(4), 'big', signed=False) if args.random \
		else 42
	print(f'{seed=}')
	torch.manual_seed(seed)
	# torch.set_printoptions() # Here just to remember me that it exists.

	device = torch.device(args.device)

	if args.data == 'mnist':
		n_rows = MNIST_N_ROWS
		n_cols = MNIST_N_COLS
		n_channels = MNIST_N_CHANNELS
		n_pixels = n_rows*n_cols*n_channels
		n_classes = MNIST_N_CLASSES

		images_data = b''
		for file in ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']:
			with download_or_read(MNIST_BASE_URL, file) as f:
				buf = memoryview(gzip.open(f).read())
				images_data += read_mnist_images(buf)

		labels_data = b''
		for file in ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
			with download_or_read(MNIST_BASE_URL, file) as f:
				buf = memoryview(gzip.open(f).read())
				labels_data += read_mnist_labels(buf)
		del buf, f, file

		with warnings.catch_warnings():
			warnings.simplefilter('ignore', UserWarning)
			images = torch.frombuffer(images_data, dtype=torch.uint8).float()
			del images_data
			labels = torch.frombuffer(labels_data, dtype=torch.uint8).long()
			del labels_data

		if args.peek:
			write_ppm('peek.ppm', images[0:n_rows*n_cols].long(),
				n_rows, n_cols)

		images = images.div(255).view(
			(MNIST_N_TRAIN+MNIST_N_TEST, n_channels, n_rows, n_cols)
		).to(device)
		labels = labels.to(device)

		train_images = images[0:MNIST_N_TRAIN]
		train_labels = labels[0:MNIST_N_TRAIN]
		test_images = images[MNIST_N_TRAIN:]
		test_labels = labels[MNIST_N_TRAIN:]
	elif args.data == 'cifar':
		n_rows = CIFAR_N_ROWS
		n_cols = CIFAR_N_COLS
		n_channels = CIFAR_N_CHANNELS
		n_pixels = n_rows*n_cols*n_channels
		n_classes = CIFAR_N_CLASSES

		with download_or_read(CIFAR_BASE_URL, 'cifar-10-binary.tar.gz') as f:
			tar = tarfile.open(None, 'r:gz', f)
			data = b''
			# TODO: this can be made three times faster if we use tar.next()
			for member_name in [
				'cifar-10-batches-bin/test_batch.bin',
				'cifar-10-batches-bin/data_batch_1.bin',
				'cifar-10-batches-bin/data_batch_2.bin',
				'cifar-10-batches-bin/data_batch_3.bin',
				'cifar-10-batches-bin/data_batch_4.bin',
				'cifar-10-batches-bin/data_batch_5.bin',
			]:
				data += tar.extractfile(member_name).read()

		bytes_in_label = 1
		bytes_in_row = n_pixels + bytes_in_label
		if len(data) % bytes_in_row:
			raise Exception('Bad CIFAR file.')

		rows = len(data)//bytes_in_row
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', UserWarning)
			tensor = torch.frombuffer(data, dtype=torch.uint8)
			del data
		labels = torch.as_strided(tensor, (rows,), (bytes_in_row,)).long()
		images = torch.as_strided(tensor, (rows, n_pixels),
			(bytes_in_row, bytes_in_label), bytes_in_label).float()

		if args.peek:
			write_ppm('peek.ppm', images[2][0:n_rows*n_cols].long(),
				n_rows, n_cols)

		if (labels >= 10).any():
			raise Exception('Bad CIFAR file.')

		labels = labels.to(device)
		images = images.div(255).view(rows, n_channels, -1)
		images = images.sub(images.mean(dim=2).unsqueeze(dim=2)) \
			.div(images.std(dim=2).unsqueeze(dim=2)) \
			.view(rows, n_channels, n_rows, n_cols).to(device)

		test_images = images[0:CIFAR_N_TEST]
		test_labels = labels[0:CIFAR_N_TEST]
		train_images = images[CIFAR_N_TEST:]
		train_labels = labels[CIFAR_N_TEST:]
	else:
		assert False

	train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
	test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

	train_dataloader = torch.utils.data.DataLoader(train_dataset,
		batch_size=args.batch, shuffle=True)
	test_dataloader = torch.utils.data.DataLoader(test_dataset,
		batch_size=args.batch, shuffle=False)

	if args.model == 'fc':
		logit_normalizer = torch.nn.Softmax(dim=1)
		label_postproc = OneHot(n_classes)
		criterion = torch.nn.MSELoss()
		width = 200
		depth = 1
		if args.small:
			if args.data == 'mnist':
				net = make_fcnet(0, n_pixels, 3, 0, n_classes).to(device)
			else:
				net = make_fcnet(0, n_pixels, 15, 1, n_classes).to(device)
		else:
			net = make_fcnet(args.intr, n_pixels, width, depth, n_classes).to(device)
		del width, depth
	elif args.model == 'lenet':
		logit_normalizer = torch.nn.Identity()
		label_postproc = torch.nn.Identity()
		criterion = torch.nn.CrossEntropyLoss()
		if args.small:
			net = make_small_lenet(n_channels, n_classes, n_rows, n_cols).to(device)
		else:
			net = make_lenet(args.intr, n_channels, n_classes, n_rows, n_cols).to(device)
	else:
		assert False

	if args.load:
		net = torch.load('model.pt')

	optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
	print(net)
	n_parameters = count_parameters(net)
	print(f'{n_parameters=}')

	if args.dry:
		return 0

	best_accuracy = train(
		net,
		args.epochs,
		args.patience,
		train_dataloader,
		logit_normalizer,
		label_postproc,
		criterion,
		optimizer,
		test_dataloader
	)

	torch.save(net, 'model.pt')

	print('\a', end='')
	return 0

if __name__ == '__main__':
	raise SystemExit(main())

# def generate_arguments():
# 	coefficients = [2, 1.5, 0.5, 0.1, 0.05, 0.01]

# 	return (
# 		(net, epochs, patience, train_dataloader, logit_normalizer,
# 		label_postproc, criterion,
# 		torch.optim.SGD(net.parameters(), lr=i), test_dataloader)
# 		for i in coefficients
# 	)
# import gc; gc.freeze() # NOTE: Is this always useful?
# with multiprocessing.Pool() as pool:
# 	res = pool.starmap(train, generate_arguments())
# print(res)
