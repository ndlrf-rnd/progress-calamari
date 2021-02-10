import glob
import math
import os
import pathlib
import re
import numpy
import random

from sklearn.utils.random import sample_without_replacement

from PIL import Image
import texturize
import tqdm
import torchvision
import time
import datetime

DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 64
DEFAULT_LIMIT = -1
REMAKE_VARIATIONS = 1
REMIX_VARIATIONS = 1
DEFAULT_PRECISION = 'float32'


def sample_comb(items, nsamp=100):
    dims = (len(items), len(items))
    idx = sample_without_replacement(numpy.prod(dims), nsamp)
    res = numpy.vstack(numpy.unravel_index(idx, dims)).T
    return tuple([
        tuple(items[res[i, j]] for j in range(res.shape[1]))
        for i in range(res.shape[0])
    ])


def random_crop(image, w, h):
    x = random.randint(0, max(0, image.size[0] - w))
    y = random.randint(0, max(0, image.size[1] - h))
    cropped_im = image.crop(
        (
            x,
            y,
            x + min(image.size[0], w),
            y + min(image.size[1], h),
        )
    )
    return img_utils.tile_image(cropped_im, w, h)


def get_fn(fn):
    return os.path.splitext(os.path.basename(fn))[0]


def save_as_next(output_dir, res_im, prefix=None):
    ts = int(time.mktime(datetime.datetime.today().timetuple()))
    op = pathlib.Path(output_dir) / ('{}{:06d}.png'.format('' if prefix is None else '{}.'.format(prefix), ts))
    os.makedirs(op.parent, exist_ok=True)
    torchvision.transforms.ToPILImage()(res_im).convert("RGB").save(op)
    return op


def generate_backgrounds(
        input_glob='input/*.*',
        output_dir='output',
        samples=256,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        limit=DEFAULT_LIMIT,
        original=True,
        precision=DEFAULT_PRECISION,
        remix_variations=REMIX_VARIATIONS,
        remake_variations=REMAKE_VARIATIONS,
        **kwargs,
):
    input_paths = [pathlib.Path(p) for p in glob.glob(input_glob) if pathlib.Path(p).is_file()]
    combos = sample_comb(input_paths, samples)

    for sid, combo in enumerate(tqdm.tqdm(combos)):
        #     sources = [
        #       random_crop(
        #         Image.open(ip).convert('RGB'),
        #         max(width, height),
        #         max(width, height),
        #       )
        #       for ip in combo
        #     ]

        if original:
            res = save_as_next(
                output_dir,
                numpy.array(random_crop(Image.open(str(combo[0])).convert('RGB'), width, height)),
                get_fn(combo[0])
            )
            print('{} -[Fittng]-> {} ({} x {})'.format(combo[0], res, width, height))
            yield res

        if (remake_variations is None) or (remake_variations > 0):
            remake = texturize.commands.Remake(*[
                random_crop(Image.open(str(im_path)).convert('RGB'), width, height)
                for im_path in combo
            ])

            for octave, result_remake in enumerate(
                    texturize.api.process_octaves(remake, size=(width, height), variations=remake_variations,
                                                  precision=precision, **kwargs)
            ):
                print('[Remake] {} Octave {}'.format(' + '.join([str(c) for c in combo]), octave + 1))

            for im in result_remake.images:
                res = save_as_next(output_dir, im.cpu(), '{}.remake.{}'.format(get_fn(combo[0]), get_fn(combo[1])))
                print('[Remake] {} -> {} ({} x {})'.format(' + '.join([str(c) for c in combo]), res, width, height))
                yield res

        if (remix_variations is None) or (remix_variations > 0):
            remix = texturize.commands.Remix(random_crop(Image.open(str(combo[0])).convert('RGB'), width, height))
            for octave, result_remix in enumerate(
                    texturize.api.process_octaves(remix, size=(width, height), variations=remix_variations,
                                                  precision=precision, **kwargs)
            ):
                print('[Remix] {} Octave '.format(str(combo[0])), octave + 1)

            for im in result_remix.images:
                res = save_as_next(output_dir, im.cpu(), '{}.remix'.format(get_fn(combo[0])))
                print('[Remix] {} -> {} ({} x {})'.format(str(combo[0]), res, width, height))
                yield res
