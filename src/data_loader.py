from ucimlrepo import fetch_ucirepo


def load_bank_marketing_data():
    """
    Load the UCI Bank Marketing dataset.

    Returns:
        X: features
        y: target
    """
    bank_marketing = fetch_ucirepo(id=222)

    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    return X, y