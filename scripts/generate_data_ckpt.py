from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')

from libs.data.mass import Mass


if __name__ == '__main__':
    dataset = Mass(load_checkpoint=False)
    dataset.save_checkpoint()
    print('Checking saved checkpoint')
    del dataset
    dataset = Mass(load_checkpoint=True)
    print('We are good!')

