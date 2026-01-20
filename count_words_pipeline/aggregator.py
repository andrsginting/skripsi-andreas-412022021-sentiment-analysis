def compute_global_average(avg_per_file_dict):
    """
    Compute global average of word count across all files.
    Input: dict {filename: avg_word_count}
    """
    values = list(avg_per_file_dict.values())
    if len(values) == 0:
        return 0
    return sum(values) / len(values)
