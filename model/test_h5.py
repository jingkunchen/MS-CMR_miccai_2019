import h5py
from keras import layers

def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("      {}: {}".format(key, value))  

            print("    Dataset:")
            for name, d in g.items(): # 读取各层储存具体信息的Dataset类
                print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
                print("      {}: {}".format(name. d.value))
    finally:
        f.close()
path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/new/"+"arg_adversarial_weights_epoch_20_8_9_8.h5"
# print_keras_wegiths(path)
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)