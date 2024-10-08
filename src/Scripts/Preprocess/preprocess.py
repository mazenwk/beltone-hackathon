from .data_preprocess.intraday_gold_price_preprocess import IntradayGoldPricesPreprocessor


class Preprocessor:
    def __init__(self):
        ...

    @staticmethod
    def preprocess_intraday_gold_prices(raw_df):
        gpp = IntradayGoldPricesPreprocessor(raw_df)
        intra_df = gpp.preprocess()
        return intra_df
