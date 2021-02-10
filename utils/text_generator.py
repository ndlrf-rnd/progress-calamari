import glob
import numpy
import os
import io
import re


def get_text_generator(
    texts_glob=None,
    alphabet='',
    text_length=(32, 256,),
):
    """Generates strings of sentences using only the letters in alphabet.

    Args:
        texts_glob: path to text files
        alphabet: The alphabet of permitted characters
        text_length: The maximum length of the string
    """
    lines = []
    min_text_length, max_text_length = text_length
    if texts_glob is not None:
        for f in glob.glob(texts_glob):
            if os.path.isfile(f):
                with io.open(f, encoding='utf-8') as fd:
                    lines += [
                        l[:max_text_length] if ((max_text_length is not None) and (max_text_length > 0)) else l
                        for l in fd.read().split('\n')
                        if len(l.strip()) >= min_text_length
                    ]
    numpy.random.shuffle(lines)

    if len(lines) > 0:
        cur = 0
        while True:
            sentence = re.sub(
                r'Ё',
                r'Е',
                re.sub(
                    r'ё',
                    r'е',
                    lines[cur % len(lines)]
                )
            ).strip()
            sentence = ''.join([s for s in sentence if ((alphabet is None) or (len(alphabet) == 0) or (s in alphabet))])
            sentence = sentence.strip()
            cur += 1
            if len(sentence) <= min_text_length:
                continue
            spaces = [0] + [
                m.start() + 1 for m in re.finditer(' ', sentence)
                if (m.start() + 1 + min_text_length) <= len(sentence)
            ]
            sentence = sentence[numpy.random.choice(spaces):]
            yield sentence
    else:
        gen = essential_generators.DocumentGenerator()
        while True:
            sentence = gen.sentence()
            sentence = ''.join([s for s in sentence if (alphabet is None or s in alphabet)])
            if max_text_length is not None:
                sentence = sentence[:max_text_length]
            yield sentence

