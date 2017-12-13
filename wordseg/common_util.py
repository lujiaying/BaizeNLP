def get_word_states(word):
    """
    Args:
        word: string
    Returns:
        word_states: list, ['B', 'E']
    """
    word_states = []
    if len(word) == 1:
        word_states.append('S')
    else:
        for idx, c in enumerate(word):
            if idx == 0:
                word_states.append('B')
            elif idx == len(word) - 1:
                word_states.append('E')
            else:
                word_states.append('M')
    return word_states

def convert_path2segments(sentence, path):
    """
    Args:
        sentence: string
        path: list, len(sentence) EQUALS TO len(path)
    Returns:
        segments: list of strings, elements are tokenized terms
    """
    segments = []
    seg = ''

    if len(sentence) != len(path):
        return segments

    for idx in range(len(path)):
        seg += sentence[idx]

        if path[idx] == 'S' or path[idx] == 'E':
            segments.append(seg)
            seg = ''
    return segments
