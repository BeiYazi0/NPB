import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    from utils import Tester

    # Tester.test_masked_resnet18_muti_npb(path=sys.argv[2])
    # Tester.test_masked_vgg19_muti_npb(path=sys.argv[2])
    Tester.test_masked_resnet20_muti_npb(path=sys.argv[2])