import argparse
import pprint
from collections import OrderedDict


class TestOptions:

    def __init__(self):

        parser = argparse.ArgumentParser()

        # ------------------
        # dataset parameters
        # ------------------
        parser.add_argument('--pre_trained', type=str,
                            default='snapshots/places2/partial_extractor/places2.pt')

        parser.add_argument('--image_root', type=str,
                            default='../basic_code/testsets/places2_test')#places2test,CELEBAHQtest,parisstreet_test
        parser.add_argument('--mask_root', type=str,
                            default='../basic_code/testsets/testmask(0.3_0.4)')

        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--load_size', type=int, default=(256, 256))
        parser.add_argument('--sigma', type=float, default=2.)
        parser.add_argument('--mode', type=str, default='test')

        parser.add_argument('--result_root', type=str, default='results/')
        parser.add_argument('--number_eval', type=int, default=1200)

        self.opts = parser.parse_args()

    @property
    def parse(self):

        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts
