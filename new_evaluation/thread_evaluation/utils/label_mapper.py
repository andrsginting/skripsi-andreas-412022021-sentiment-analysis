def map_score_to_label(score, pos_th=0.05, neg_th=-0.05):
    """
    Mengubah skor sentimen kontinu (-1 s/d 1)
    menjadi kelas diskret: positif / netral / negatif
    """
    if score > pos_th:
        return "positif"
    elif score < neg_th:
        return "negatif"
    else:
        return "netral"
