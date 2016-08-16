import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('virt_packets.hdf5','r')

# read and sort
nus = f['/virt_packet_nus/values'][:]
en = f['/virt_packet_energies/values']

# indices to order nu
indices = np.argsort(nus)
nus, en = nus[indices], en[indices]


h, bins = np.histogram(nus, bins=10000)
plt.plot(h)

sl = slice(1000, 1002)
print(bins[sl])
print(h[sl.start])

# find range of packets in bin
index_range = np.searchsorted(nus, bins[sl])
en_in_bin = en[index_range[0]:index_range[1]]

# distribution of sum of elements, skip last element if not the sum of n elements
n = 15
sum_samples = np.add.reduceat(samples, np.arange(0, len(samples), n))
if len(samples) % n != 0:
    sum_samples = sum_samples[:-1]
plt.clf(); plt.hist(sum_samples)

# first and second moment
first = np.mean(samples)
second = np.sum(samples * samples) / len(samples)
