import os

if __name__ == '__main__':

    cores = 8
    data_dir = '/home/ubuntu/clean'
    output = '/home/ubuntu/artifacts'
    images = '/home/ubuntu/33k-images/lists/good_images_400.csv'
    backbones = ['ResNet50', 'InceptionV3', 'Xception', 'NASNet']
    poolings = ['avg', 'max']
    
    for backbone in backbones:
        for pooling in poolings:
            command = "python3 main.py --data_dir={} --output_dir={} --images={} --backbone={} --pooling={} --cores={}".format(data_dir, output, images, backbone, pooling, cores)
            os.system(command)
