import tarfile
import timeit

def f():
	with tarfile.open('cifar-10-binary.tar.gz', 'r:gz') as tar:
		data = b''
		for member_name in [
			'cifar-10-batches-bin/test_batch.bin',
			'cifar-10-batches-bin/data_batch_1.bin',
			'cifar-10-batches-bin/data_batch_2.bin',
			'cifar-10-batches-bin/data_batch_3.bin',
			'cifar-10-batches-bin/data_batch_4.bin',
			'cifar-10-batches-bin/data_batch_5.bin',
		]:
			data += tar.extractfile(member_name).read()

def g():
	with tarfile.open('cifar-10-binary.tar.gz', 'r:gz') as tar:
		data = b''
		while (member := tar.next()):
			if member.name in [
				'cifar-10-batches-bin/test_batch.bin',
				'cifar-10-batches-bin/data_batch_1.bin',
				'cifar-10-batches-bin/data_batch_2.bin',
				'cifar-10-batches-bin/data_batch_3.bin',
				'cifar-10-batches-bin/data_batch_4.bin',
				'cifar-10-batches-bin/data_batch_5.bin',
			]:
				data += tar.extractfile(member).read()

if __name__ == '__main__':
	print(timeit.timeit(f, number=10))
	print(timeit.timeit(g, number=10)) # WINS!
