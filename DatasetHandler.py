import struct
import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def apply_pixel_intensity_threshold(dataset, threshold = 75.0):
    threshold *= (255.0 / 100.0)
    return dataset >= threshold

def reshape_dataset(raw_dataset):
    return np.squeeze([sample.reshape(-1, 1) for sample in raw_dataset], axis = 2)

def divide_into_batches(samples, batch_size = 600):
    return [samples[i * batch_size : i+1 * batch_size] for i in range(int((samples.shape[0]/batch_size)))]
    