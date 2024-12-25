import numpy as np
import os
import tensorflow as tf



def main():
    ae_path = './autoencoder_3072.npy'
    data_dict = np.load(ae_path, allow_pickle=True, encoding='latin1').item()
    print()
    pass


if __name__ == '__main__':
    main()

