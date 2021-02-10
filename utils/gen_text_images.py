import os
import math
import glob
import typing
import random
import zipfile
import string
import cv2
import tqdm
import numpy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import skimage
import warnings
import numpy
import pylab
import scipy.ndimage

from . import img_utils

"""
Functions below were taken from ocrodeg project:
https://github.com/NVlabs/ocrodeg/
"""

def percent_black(image):
    n = prod(image.shape)
    k = sum(image < 0.5)
    return k * 100.0 / n

def binary_blur(image, sigma, noise=0.0):
    p = percent_black(image)
    blurred = scipy.ndimage.gaussian_filter(image, sigma)
    if noise > 0:
        blurred += pylab.randn(*blurred.shape) * noise
    t = percentile(blurred, p)
    return array(blurred > t, 'f')

# multiscale noise

def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h/scale+1), int(w/scale+1)
    data = pylab.rand(h0, w0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scipy.ndimage.zoom(data, scale)
    return result[:h, :w]

def make_multiscale_noise(shape, scales, weights=None, limits=(0.0, 1.0)):
    if weights is None: weights = [1.0] * len(scales)
    result = make_noise_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_noise_at_scale(shape, s) * w
    lo, hi = limits
    result -= amin(result)
    result /= amax(result)
    result *= (hi-lo)
    result += lo
    return result

def make_multiscale_noise_uniform(shape, srange=(1.0, 100.0), nscales=4, limits=(0.0, 1.0)):
    lo, hi = log10(srange[0]), log10(srange[1])
    scales = numpy.random.uniform(size=nscales)
    scales = add.accumulate(scales)
    scales -= amin(scales)
    scales /= amax(scales)
    scales *= hi-lo
    scales += lo
    scales = 10**scales
    weights = 2.0 * numpy.random.uniform(size=nscales)
    return make_multiscale_noise(shape, scales, weights=weights, limits=limits)

# random blobs

def random_blobs(shape, blobdensity, size, roughness=2.0):
    h, w = shape
    numblobs = int(blobdensity * w * h)
    mask = numpy.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[numpy.random.randint(0, h - 1), numpy.random.randint(0, w - 1)] = 1
    dt = scipy.ndimage.distance_transform_edt(1-mask)
    mask =  numpy.array(dt < size, 'f')
    mask = scipy.ndimage.gaussian_filter(mask, size / (2 * roughness ) if roughness else size)
    mask -= numpy.amin(mask)
    mask /= numpy.amax(mask)
    noise = pylab.rand(h, w)
    res = numpy.array(mask * noise > 0.5, 'uint8').astype('uint8')
    return res

def random_blotches(image, fgblobs, bgblobs, fgscale=10, bgscale=10):
  fg = random_blobs(image.shape[:2], fgblobs, fgscale)
  bg = random_blobs(image.shape[:2], bgblobs, bgscale)
  image = minimum(maximum(image[:,:], fg), 1 - bg)
  return image



LIGATURES = {'\U0000FB01': 'fi', '\U0000FB02': 'fl'}
LIGATURE_STRING = ''.join(LIGATURES.keys())

def resolve_random(scalar_or_tuple):
  """Resolve randrange tuple or return value back in case of being not a tuple.
  """
  if isinstance(scalar_or_tuple, tuple):
    return  numpy.random.uniform(low=scalar_or_tuple[0], high=scalar_or_tuple[1])
  return scalar_or_tuple


def wh2c(width, height):
      return numpy.array([
        numpy.array([
          [0, 0],
          [width, 0],
          [width, height],
          [0, height]
        ]).astype('float32')
      ])

def get_image_font_px(path, size_px, text='|', debug=False):
  if path is not None:
    # Fixme: works pretty slow due font rendering brute forcing
    # starting font size considering 1pt = 1px
    current_size_pt = int(size_px)

    font = PIL.ImageFont.truetype(path, current_size_pt)
    current_size_px = font.getsize(text)[1]
    increment = 1 if (current_size_px < size_px) else -1

    while (current_size_px < size_px) if (increment > 0) else (current_size_px > size_px):
      current_size_pt += increment
      font = PIL.ImageFont.truetype(path, current_size_pt)
      current_size_px = font.getsize(text)[1]

    if (current_size_px > size_px):
      current_size_pt -= 1
      font = PIL.ImageFont.truetype(path, current_size_pt)
      current_size_px = font.getsize(text)[1]

    if debug:
      print(
        'Font {} will be rendered as {}pt ({}px height) to fit {}px height constraint'.format(
          path,
          current_size_pt,
          current_size_px,
          size_px
        )
      )

    return font
  else:
    print('Font {} not found'.format(path))
    return PIL.ImageFont.load_default()

def draw_text_image(
  text,
  font_size_px,
  height,
  width,
  font_path,
  skew_x=0,
  skew_y=0,
  color=(0, 0, 0),
  permitted_contour=None,
  debug=False
):
    """Get a transparent image containing text.

    Args:
        text: The text to draw on the image
        font_size_px: The size of text to show.
        height: The height of the output image
        width: The width of the output image
        fonts: A dictionary of {subalphabet: paths_to_font}
        skew_x: skew ratio (base = font size) about the X axis
        skew_y: skew ratio (base = font size) about the Y axis
        color: The color of drawn text

    Returns:
        An (image, lines) tuple where image is the
        transparent text image and lines is a list of lines
        where each line itself is a list of (box, character) tuples and
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    font = get_image_font_px(font_path, size_px=font_size_px, text=text, debug=debug)

    x = 0
    y = 0
    margin_x = font_size_px // 2
    margin_y = font_size_px // 4

    text = text.strip()
    space_offsets = []
    for idx in range(len(text)):
      character = text[idx]
      (character_width, character_height), (offset_x, offset_y) = font.font.getsize(character)

      if character in LIGATURES:
          dx = character_width / len(LIGATURES[character])
      else:
        dx = character_width
      if character == ' ':
        space_offsets.append(idx)

      if (x + dx) >= width:
        if len(space_offsets) > 0:
          if debug:
            print(
              'Reducing length of "{}" {} -> "{}" {}'.format(
                text,
                len(text),
                space_offsets[-1],
                text[space_offsets[-1]]
              )
            )
          text = text[:space_offsets[-1]]
        break

      x += character_width + offset_x
      y += character_height + offset_y

    text_width, text_height = font.getsize(text)
    text_width += margin_x * 2
    text_height += margin_y * 2
    text_contour = wh2c(text_width, text_height)
    transform_base = text_height

    skew1_x_pt = transform_base * (skew_x / 2) if skew_y >= 0 else 0
    skew1_y_pt = transform_base * (skew_y / 2) if skew_y >= 0 else 0
    skew2_x_pt = transform_base * (skew_x / 2) if skew_x < 0 else 0
    skew2_y_pt = transform_base * (skew_y / 2) if skew_y < 0 else 0

    transformed_contour = numpy.array([
      numpy.float32(
        [
          [
            0 + skew1_x_pt,
            0 + skew2_y_pt
          ],
          [
            text_width - skew1_x_pt,
            0 + skew1_y_pt
          ],
          [
            text_width - skew2_x_pt,
            text_height - skew1_y_pt
          ],
          [
            0 + skew2_x_pt,
            text_height - skew2_y_pt
          ]
        ]
      )
    ])
    M = cv2.getPerspectiveTransform(src=text_contour[0,:,:], dst=transformed_contour[0,:,:])

    text_contour = text_contour.astype('int32')
    transformed_contour = transformed_contour.astype('int32')

    min_x = transformed_contour[:,:,0].min()
    max_x = transformed_contour[:,:,0].max()

    min_y = transformed_contour[:,:,1].min()
    max_y = transformed_contour[:,:,1].max()

    image = PIL.Image.new(mode='RGBA', size=(max_x - min_x, max_y - min_y), color=(255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(margin_x, margin_y), text=text, fill=color + (255, ), font=font) # anchor="la",
    image = numpy.array(image)
    image = cv2.warpPerspective(src=numpy.array(image), M=M, dsize=(max_x, max_y))

    if debug:
      image = cv2.drawContours(
        image,
        contours=text_contour,
        contourIdx=0,
        color=(0, 255, 0, 128),
        thickness=int(1)
      )
      image = cv2.drawContours(
        image,
        contours=transformed_contour,
        contourIdx=0,
        color=(255, 0, 0, 128),
        thickness=int(1)
      )
    try:
      return image, transformed_contour, text
    except Exception as e:
      raise Exception('ERROR during rendering string: "{}"'.format(text))


def grid_distortion(
    img,
    num_steps=8,
    distort_limit=0.5,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REPLICATE,
    value=None,
):
    """
    Perform a grid distortion of an input image.
    Reference:
      - http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    Reference source:
      - https://github.com/albumentations-team/albumentations/blob/ff83de8a51184bca97be3a9fc6905c56b44c494a/albumentations/augmentations/functional.py#L899
    """
    stepsx = numpy.random.dirichlet(numpy.ones(num_steps), size=1)[0]
    stepsx = (stepsx / (stepsx.max() - stepsx.min())) * max(0, min(1, distort_limit))
    stepsx += 1 - stepsx.mean()

    stepsy = numpy.random.dirichlet(numpy.ones(num_steps), size=1)[0]
    stepsy = (stepsy / (stepsy.max() - stepsy.min())) * max(0, min(1, distort_limit))
    stepsy += 1 - stepsy.mean()


    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = numpy.zeros(width, numpy.float32)
    prev = 0
    for idx in range(num_steps):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * stepsx[idx]

        xx[start:end] = numpy.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = numpy.zeros(height, numpy.float32)
    prev = 0
    for idx in range(num_steps):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * stepsy[idx]

        yy[start:end] = numpy.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = numpy.meshgrid(xx, yy)
    map_x = map_x.astype(numpy.float32)
    map_y = map_y.astype(numpy.float32)

    return cv2.remap(
      img,
      map1=map_x,
      map2=map_y,
      interpolation=interpolation,
      borderMode=border_mode,
      borderValue=value
    )

def crossfade(foreground, alpha, background):
  if len(foreground.shape) == 2:
    foreground = numpy.stack([foreground] * background.shape[2], axis=2)
  if len(alpha.shape) == 2:
    alpha = numpy.stack([alpha] * background.shape[2], axis=2)
  alpha = alpha.astype(numpy.float32) / 256
  assert alpha.min() >= 0
  assert alpha.max() <= 1
  weighted_foreground = alpha * foreground
  weighted_background = (1.0 - alpha) * background
  return (weighted_foreground + weighted_background).astype(numpy.uint8)

def get_image_generator(
  height,
  width,
  font_paths,
  text_generator,
  font_size_px: typing.Union[int, typing.Tuple[int, int]] = 18,
  backgrounds: typing.List[typing.Union[str, numpy.ndarray]] = None,
  decals: typing.List[typing.Union[str, numpy.ndarray]] = None,
  background_crop_mode='crop',
  skew_x: typing.Union[float, typing.Tuple[float, float]] = 0,
  skew_y: typing.Union[float, typing.Tuple[float, float]] = 0,
  distort_limit: typing.Union[float, typing.Tuple[float, float]] = 0,
  decals_count=(0, 1),
  decals_opacity=(0.25, 1.0,),
  decals_scale=(0.1, 2.0),
  decals_rotate=(0.0, math.pi*2),
  blobs_opacity=0.5,
  margin=0,
  debug=False,
):
    """Create a generator for images containing text.

    Args:
        height: The height of the generated image
        width: The width of the generated image.
        font_paths: path to fonts [path_to_font1, path_to_font2]
        text_generator: See пу
        font_size: The font size to use. Alternative, supply a tuple
            and the font size will be randomly selected between
            the two values.
        backgrounds: A list of paths to image backgrounds or actual images
            as numpy arrays with channels in RGB order.
        background_crop_mode: One of letterbox or crop, indicates
            how backgrounds will be resized to fit on the canvas.
        skew_x: The X-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        skew_y: The Y-axis text rotation to use. Alternative, supply a tuple
            and the rotation will be randomly selected between
            the two values.
        margin: The minimum margin around the edge of the image.
        background_augmenter: An image augmenter to be applied to backgrounds
        foreground_augmenter: An image augmenter to be applied to text
        debug: Draw the permitted contour onto images

    Yields:
        Tuples of (image, lines) where image is the
        transparent text image and lines is a list of lines
        where each line itself is a list of (box, character) tuples and
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    decal_idx = 0
    for idx, text in enumerate(text_generator):
        if (idx % len(backgrounds)) == 0:
            random.shuffle(backgrounds)
        current_background = PIL.Image.open(backgrounds[idx % len(backgrounds)]).convert('RGB')


        if (idx % len(font_paths)) == 0:
            random.shuffle(font_paths)
        font_path = font_paths[idx % len(font_paths)]


        current_font_size_px = resolve_random(font_size_px)

        text_color = tuple(numpy.random.randint(low=0, high=50, size=3))
        current_skew_x = resolve_random(skew_x)
        current_skew_y = resolve_random(skew_y)
        current_distort_limit = resolve_random(distort_limit)
        # Don't resolve distort limit!
        text_image, contour, text = draw_text_image(
          text=text,
          width=width,
          height=height,
          font_size_px=current_font_size_px,
          font_path=font_path,
          skew_x=current_skew_x,
          skew_y=current_skew_y,
          color=text_color,
          debug=debug
        )

        h, w = [dim + margin * 2 for dim in text_image.shape[0:2]]

        image = numpy.array(img_utils.tile_image(current_background, target_w=w, target_h=h))

        fg_blobs = random_blobs(text_image.shape[:2], 3e-3, 20)
        bg_blobs = random_blobs(text_image.shape[:2], 3e-4, 20)
        current_decals_count = min(resolve_random(decals_count), len(decals))
        current_decal_degrade = bool(random.getrandbits(1))
        text_alpha = numpy.minimum(text_image[:, :, 3], 256 - (bg_blobs * blobs_opacity))
        tx1 = margin
        ty1 = margin
        tx2 = margin + text_image.shape[0]
        ty2 = margin + text_image.shape[1]
        decal_opacity = resolve_random(decals_opacity)

        if current_decals_count > 0:
          decals_alpha_image = numpy.zeros(
            shape=(image.shape[0], image.shape[1],),
            dtype = image.dtype
          )

          for idx in numpy.arange(current_decals_count):

            current_decal_rotate = resolve_random(decals_rotate)
            current_decal_scale = resolve_random(decals_scale)

            decal_bitmap_or_path = decals[decal_idx % len(decals)]
            if isinstance(decal_bitmap_or_path, str):
              current_decal = PIL.Image.open(decal_bitmap_or_path)
            else:
              current_decal = decal_bitmap_or_path

            if decal_idx >= len(decals):
              random.shuffle(decals)
              decal_idx = 0
            else:
              decal_idx += 1

            td = current_decal.copy()
            td = (skimage.transform.rotate(td, current_decal_rotate, resize=True) * 255).astype(numpy.uint8)
            td = (skimage.transform.rescale(td, current_decal_scale) * 255).astype(numpy.uint8)

            offset_x = numpy.random.uniform(low=0, high=max(0, decals_alpha_image.shape[0] - td.shape[0]))
            offset_y = numpy.random.uniform(low=0, high=max(0, decals_alpha_image.shape[1] - td.shape[1]))

            td = td[
              0:min(decals_alpha_image.shape[0], td.shape[0]),
              0:min(decals_alpha_image.shape[1], td.shape[1]),
            ]

            x1 = int(offset_x)
            y1 = int(offset_y)
            x2 = int(offset_x + td.shape[0])
            y2 = int(offset_y + td.shape[1])

            decals_alpha_image[x1:x2,y1:y2] = numpy.maximum(td, decals_alpha_image[x1:x2, y1:y2])
            if current_decal_degrade:
              main_text_alpha = text_alpha * (1 - (decals_alpha_image[tx1:tx2,ty1:ty2] / 255))
              decal_text_alpha = text_alpha * (decals_alpha_image[tx1:tx2,ty1:ty2] / 255) * decal_opacity
              text_alpha = main_text_alpha + decal_text_alpha
            else:
              image = crossfade(
                image * (1 - decal_opacity),
                decals_alpha_image,
                image,
              )


        image[tx1:tx2,ty1:ty2,0:3] = crossfade(
          text_image[:,:,:3],
          text_alpha,
          image[tx1:tx2,ty1:ty2,0:3]
        )
        image = grid_distortion(image, distort_limit=current_distort_limit)
        yield image, contour, text, font_path
