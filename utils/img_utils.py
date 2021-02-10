from collections.abc import Iterable
import json
import io
import os
import numpy
import math
import stat
import PIL.Image


def get_geom_wh(geom):
    if len(geom.bounds) == 4:
        return (
            geom.bounds[2] - geom.bounds[0],
            geom.bounds[3] - geom.bounds[1],
        )
    else:
        return 0, 0


def get_block_wh(block):
    if len(block['geom'].bounds) == 4:
        return (
            block['geom'].bounds[2] - block['geom'].bounds[0],
            block['geom'].bounds[3] - block['geom'].bounds[1],
        )
    else:
        return 0, 0


def is_horisontal(geom):
    box = geom.bounds if hasattr(geom, 'bounds') else [
        geom[0][0],
        geom[0][1],
        geom[-1][0],
        geom[-1][1],
    ]
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    return w >= h


def flatten(iterable):
    for el in iterable:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def omit(list_of_dicts, fields):
    if isinstance(fields, basestring):
        fields = {fields}
    return [
        {
            k: v
            for k, v in d.items()
            if k not in fields
        }
        for d in list_of_dicts
    ]


def scale_to_area(width, height, target_area):
    ratio = height / width

    a = (target_area / ratio) ** (1 / 2)

    width_new = int(a)
    height_new = int(a * ratio)
    return width_new, height_new


class NpEncoder(json.JSONEncoder):
    """
    Source: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json
    -serializable/50916741
    """

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


DEFAULT_DPI = 200
SCREEN_DPI = 96


def get_dpi(ocr_json_path, default=DEFAULT_DPI):
    dpi_path = os.path.join(os.path.dirname(ocr_json_path), '{}.dpi.txt'.format(os.path.basename(ocr_json_path)))
    if os.path.exists(dpi_path):
        with io.open(dpi_path, 'r', encoding='utf-8') as f:
            return int(f.read().split('\n')[0].split(' ')[0])
    else:
        return default


def get_segments_distance(s1, s2):
    c1 = s1['segment']['geom'].centroid.coords[0]
    c2 = s2['segment']['geom'].centroid.coords[0]
    return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))


def format_table(cells, head=None, title=None):
    title_html = ''
    if title and (title is not None):
        title_html = '<h3>{}</h3>'.format(title) if title else ''

    head_html = ''
    if head and (head is not None):
        head_html = '<thead><tr>{}</tr></thead>'.format(
            '\n'.join(['<th>{}</th>'.format(th) for th in head])
        )

    body_html = '<tbody>{}</tbody>'.format(
        '\n'.join([
            '<tr>{}</tr>'.format(
                '\n'.join([
                    '<td>{}</td>'.format(td)
                    for td in row
                ])
            )
            for row in cells
        ])
    )

    return '{}<table>{}{}</table>'.format(title_html, head_html, body_html)


def actualize_time(file_path, prev_time=None):
    if os.path.exists(file_path):
        mt = os.stat(file_path)[stat.ST_MTIME]
        return mt if prev_time is None else max(mt, prev_time)
    else:
        return prev_time


def tile_image(bg, target_w=1024, target_h=1024):
  bg_w, bg_h = bg.size
  tiled_im = PIL.Image.new('RGB', (target_w, target_h))
  w, h = tiled_im.size
  for i in range(0, w, bg_w):
      for j in range(0, h, bg_h):
          bg = PIL.Image.eval(bg, lambda x: x+(i+j)/1000)
          tiled_im.paste(bg, (i, j))

  return tiled_im
