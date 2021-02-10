import fontTools.ttLib
import PIL.ImageFont
def font_supports_alphabet(filepath, alphabet, debug=False):
    """Verify that a font contains a specific set of characters.

    Args:
        filepath: Path to fsontfile
        alphabet: A string of characters to check for.
    """
    if alphabet == '':
        return True
    font = fontTools.ttLib.TTFont(filepath)
    try:
        charaters_support = [
            ord(c) in table.cmap
            for table in font.get('cmap').tables
            for c in alphabet
        ]
        supported_cases = [supported for supported in charaters_support if supported]
        support_level = len(supported_cases) / len(charaters_support)
        if debug:
            print('DEBUG: Font {} supports given alphabet on {}%'.format(filepath, int(support_level * 100)))
        return support_level
    except Exception as e:
        print('WARNING: Font {} have invalid format'.format(filepath), e)
        return False
    font = PIL.ImageFont.truetype(filepath)
    try:
        for character in alphabet:
            font.getsize(character)
    # pylint: disable=bare-except
    except:
        return False
    return True

