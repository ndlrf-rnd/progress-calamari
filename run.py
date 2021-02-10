import tqdm
import xml.etree.ElementTree as ElementTree
import io
import os
import glob
import os
import re
import random
import string
import numpy
import math

from PIL import Image
# import matplotlib.pyplot as pyplot
import essential_generators
from torchvision import transforms

import utils
# import utils/.prepare_data# import get_zip



DEBUG = False

OPEN_FONTS_URL = 'https://github.com/google/fonts/archive/master.zip'
FONTS_DIR = './input/fonts'

OPEN_CORPORA_URL = 'http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip'
TEXTS_DIR = './input/texts'

OPEN_CORPORA_TXT_PATH = './input/texts/opencorpora.txt'
OPEN_FONTS_GLOB = './input/fonts/fonts-master/*/*/*.ttf'

FONTS_BLACKLIST = set(['AdobeBlank-Regular.ttf'])

BACKGROUNDS_INPUT_GLOB = './input/backgrounds/*.*' 
TEXTS_GLOB = os.path.join(TEXTS_DIR, '*.txt')

ALPHABET_FILE = './input/alphabet.txt'

SETS = (
    (int(1024), 'train', OPEN_FONTS_GLOB, ),
    (int(128), 'test', OPEN_FONTS_GLOB, ),
)
OUTPUT_COLOR_MODE="L"
FONT_SUPPORT_RATIO_THRESHOLD = 0.75
OUTPUT_LINES_DIR = './output/lines-train_8k-test_1k-{}/'.format(OUTPUT_COLOR_MODE.lower())

SHOW_GEN_SAMPLE_EVERY = 128
DECALS_DEBUG_GRID_COLUMNS = 16

BACKGROUND_SAMPLES = 0
BACKGROUND_WIDTH = 1024
BACKGROUND_HEIGHT = 64
BACKGROUNDS_DIR = './input/backgrounds-{}x{}/'.format(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)

DEСALS_INPUT_GLOB = './input/decals/**/*.*'
DECALS_SAMPLES = 128
DECAL_WIDTH = 512
DECAL_HEIGHT = 512
DECALS_DIR = './input/decals-{}x{}/'.format(DECAL_WIDTH, DECAL_HEIGHT)


TEXTURIZE_PRECISION = 'float16'
TEXTURIZE_ORIGINAL=False
TEXTURIZE_REMAKE_VARIATIONS = 0
TEXTURIZE_REMIX_VARIATIONS = 1

STRIPE_TEXT_LENGTH = (32, 256,)

STRIPE_MAX_SIZE_PX = (1024, 54,)

STRIPE_FONT_SIZE_PX = (32, 48,)
STRIPE_SKEW_X = (-0.5, 0.5,)
STRIPE_SKEW_Y = (-0.3, 0.3,)
STRIPE_DISTORT_LIMIT = (-0.0, 0.8, )



if not os.path.exists(FONTS_DIR):
    utils.prepare_data.get_zip(OPEN_FONTS_URL, FONTS_DIR)

if not os.path.exists(OPEN_CORPORA_TXT_PATH):
    utils.prepare_data.get_zip(OPEN_CORPORA_URL, TEXTS_DIR)
    with io.open(OPEN_CORPORA_TXT_PATH, 'w+', encoding='utf-8') as outf:
        for xml_file_path in glob.glob(os.path.join(TEXTS_DIR, '*.xml')):
            print('Parsing {} ...'.format(xml_file_path))
            tree = ElementTree.parse(xml_file_path)
            for el in tqdm.tqdm(
                tree.iter('source'), 
                desc='Writing text lines to {} ...'.format(OPEN_CORPORA_TXT_PATH)
            ):
                outf.write('{}\n'.format(el.text.strip()))

                # Backgrounds
backgrounds = list(glob.glob(os.path.join(BACKGROUNDS_DIR, '*.*')))

# Decals
decals_paths = glob.glob(DEСALS_INPUT_GLOB)
decals = [numpy.array(Image.open(p).convert('RGBA')) for p in decals_paths]
print('Decals loaded: {} from {}'.format(len(decals_paths), DEСALS_INPUT_GLOB))

# fig, axs = pyplot.subplots(
#     math.ceil(len(decals_paths) / DECALS_DEBUG_GRID_COLUMNS), DECALS_DEBUG_GRID_COLUMNS,
#     figsize=(DECALS_DEBUG_GRID_COLUMNS, ((len(decals_paths) // DECALS_DEBUG_GRID_COLUMNS) + 1))
# )
# for ar in axs:
#     for ax in ar:
#         ax.set_axis_off()
        
os.makedirs(DECALS_DIR, exist_ok=True)
decals = []
for idx, dp in enumerate(decals_paths):
    im = Image.open(dp)
    ratio = DECAL_WIDTH / max(im.size)
    new_size = [int(c * ratio) for c in im.size]
    da = numpy.array(im.resize(new_size))
    if (len(da.shape) == 2):
        res = da
    elif (len(da.shape) == 3):
        if (da.shape[2] == 4):
            if 0 < da[:,:,3].mean() < 254:
                res = da[:,:,3]
            else:
                res = da[:,:,:3].mean(axis=2)
    if 1.0 < res.mean() < 254.0:
#         axs[idx // DECALS_DEBUG_GRID_COLUMNS][idx % DECALS_DEBUG_GRID_COLUMNS].imshow(decals[-1], cmap='gray')
        op = os.path.join(DECALS_DIR, 'decal-{}.png'.format(idx + 1))
        
        transforms.ToPILImage()(res).convert("RGBA").save(op)

decals_paths = glob.glob(os.path.join(DECALS_DIR, '*.png'))
decals = [numpy.array(Image.open(p).convert('L')) for p in decals_paths]
print('Decals loaded: {} from {}'.format(len(decals_paths), DEСALS_INPUT_GLOB))

# Alphabet                
ALPHABET = string.digits + string.ascii_letters + '!?. '
if (ALPHABET_FILE is not None) and os.path.exists(ALPHABET_FILE):
    with io.open(ALPHABET_FILE, encoding='utf-8') as fd:
        ALPHABET = re.sub(r'[\n\r\t]+', r'', fd.read()) + ' '
print('Alphabet:\n{}'.format(ALPHABET))             


if (BACKGROUND_SAMPLES is not None) and (BACKGROUND_SAMPLES > 0):
    from utils import paper
    os.makedirs(BACKGROUNDS_DIR, exist_ok=True)
    for idx, genp in enumerate(paper.generate_backgrounds(
        BACKGROUNDS_INPUT_GLOB,
        output_dir=BACKGROUNDS_DIR,
        samples=BACKGROUND_SAMPLES, 
        width=BACKGROUND_WIDTH,
        height=BACKGROUND_HEIGHT,
        precision=TEXTURIZE_PRECISION,
        original=TEXTURIZE_ORIGINAL,
        remake_variations=TEXTURIZE_REMAKE_VARIATIONS,
        remix_variations=TEXTURIZE_REMIX_VARIATIONS,
    )):
      pass
#         if idx % SHOW_GEN_SAMPLE_EVERY == 0:
#             pyplot.figure(figsize=(16, 16))
#             pyplot.imshow(Image.open(genp))
#             pyplot.show()


tg = utils.get_text_generator(
    TEXTS_GLOB, 
    alphabet=ALPHABET, 
    text_length=STRIPE_TEXT_LENGTH,
)

os.makedirs(OUTPUT_LINES_DIR, exist_ok=True)
with io.open(os.path.join(OUTPUT_LINES_DIR, 'full_codec.txt'), mode='w+', encoding='utf-8') as fd:
    fd.write(ALPHABET)

for (olc, subdir_name, fonts_glob) in SETS:    
    alphabet = ALPHABET
    print('Searching fonts at {}'.format(fonts_glob))
    fonts_paths = [fp for fp in glob.glob(fonts_glob) if os.path.basename(fp) not in FONTS_BLACKLIST]
    fonts = []
    unsupported_fonts = []
    if alphabet != '':
        for filepath in tqdm.tqdm(fonts_paths, desc='Filtering fonts...'):
            if utils.font_supports_alphabet(filepath, alphabet, debug=DEBUG) > FONT_SUPPORT_RATIO_THRESHOLD:
                fonts.append(filepath)
            else:
                unsupported_fonts.append(filepath)
    print(
        'Loaded {} fonts that suport given alphabet more than on {}% characters, {} fonts were ignored'.format(
            len(fonts), 
            int(FONT_SUPPORT_RATIO_THRESHOLD * 100), 
            len(unsupported_fonts)
        )
    )
    stripe_margin_px = (STRIPE_MAX_SIZE_PX[1] - max(STRIPE_FONT_SIZE_PX)) // 2
    image_generator = utils.get_image_generator(
        width=STRIPE_MAX_SIZE_PX[0],
        height=STRIPE_MAX_SIZE_PX[1],
        margin=stripe_margin_px,
        text_generator=tg,
        font_paths=fonts,
        backgrounds=backgrounds,
        decals=decals,
        font_size_px=STRIPE_FONT_SIZE_PX,
        skew_x=STRIPE_SKEW_X,
        skew_y=STRIPE_SKEW_Y,
        debug=DEBUG,
        distort_limit=STRIPE_DISTORT_LIMIT,
        decals_count=(0, 2),
        decals_opacity=(0.25, 0.65),
        decals_rotate=(0.0, math.pi*2),
        decals_scale=(0.01, 1.0),
    )

    for idx in tqdm.tqdm(range(1, olc + 1), desc='Generating text line images,,,'):
        subdir = os.path.join(OUTPUT_LINES_DIR, subdir_name)
        os.makedirs(subdir, exist_ok=True)
        image, contour, text, font = next(image_generator)
        opi = os.path.join(subdir, '{:06d}.png'.format(idx))
        opt = os.path.join(subdir, '{:06d}.gt.txt'.format(idx))
        image = transforms.ToPILImage()(image).convert(OUTPUT_COLOR_MODE)
        image.save(opi)
        with io.open(opt, mode='w+', encoding='utf-8') as fdt:
            fdt.write(text)
            if DEBUG:
              pyplot.figure(figsize=(16, 2))
              pyplot.title('[{}/{}] {}\n "{}"\n{}\n{}\n'.format(idx, olc, font, text, opi, opt))
              pyplot.imshow(image.convert('RGB'))
              pyplot.show()