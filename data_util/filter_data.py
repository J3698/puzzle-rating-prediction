from data_to_numpy import assert_data_downloaded


def read_data():
    names = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", \
             "Popularity", "NbPlays", "Themes", "GameUrl"]

    return pd.read_csv(DATA_FILE, names = names)



