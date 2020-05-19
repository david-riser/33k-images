import os

if __name__ == '__main__':

    cores = 4
    data_dir = '/home/ubuntu/data'
    output = './artifacts'
    backbones = ['ResNet50', 'InceptionV3', 'Xception', 'NASNet']
    poolings = ['avg', 'max']
    min_samples = 320
    
    for backbone in backbones:
        for pooling in poolings:
            command = "python main_mb.py --base_dir={} --output_dir={} --min_samples={} --backbone={} --pooling={} --cores={}".format(
                data_dir, output, min_samples, backbone, pooling, cores)
            os.system(command)
