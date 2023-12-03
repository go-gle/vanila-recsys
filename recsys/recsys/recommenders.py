import pandas as pd
import abc
import typing
from sklearn.exceptions import NotFittedError


class AbcRecommender:
    def __init__(self: "PopularRecommender",
                 target_col: str,
                 rank_col: str,
                 ):
        self._rank_col = rank_col
        self._target_col = target_col
        self._fitted = False

    @abc.abstractmethod
    def fit(self: "AbcRecommender", ranks: pd.DataFrame):
        pass

    @abc.abstractmethod
    def predict(self: "AbcRecommender", book: str, max_rec: int) -> typing.List[str]:
        pass


class PopularRecommender(AbcRecommender):
    def __init__(self: "PopularRecommender", target_col: str, rank_col: str):
        super().__init__(target_col=target_col,
                         rank_col=rank_col)
        self._rank_df = None
        self._sorted_rank_df = None

    def fit(self: "PopularRecommender", rank: pd.DataFrame) -> "PopularRecommender":
        self.rank = rank
        self._sorted_rank = (self.rank.groupby(self._target_col)[self._rank_col].sum()
                             .sort_values(ascending=False).reset_index())
        self._fitted = True
        return self

    def predict(self: "PopularRecommender", book: str, max_rec: int=10) -> typing.List[str]:
        if not self._fitted:
            raise NotFittedError("Popular Recommender is not fitted")
        recs = self._sorted_rank.iloc[:max_rec][self._target_col].to_list()
        cleaned_recs = [rec for rec in recs if rec != book]
        return cleaned_recs


class CorrRecommender(PopularRecommender):
    def __init__(self: "CorrRecommender", target_col: str, rank_col: str):
        super().__init__(target_col=target_col, rank_col=rank_col)
        self._target_val_set = None


    def fit(self: "CorrRecommender", rank: pd.DataFrame) -> "CorrRecommender":
        super(CorrRecommender, self).fit(rank)
        self._target_val_set = set(rank[self._target_col].to_list())
        return self

    def _calc_corr_for_df(self: "CorrRecommender", target_index: str):
        # We can evaluate correlations only for books which were read by the same users
        rank = self.rank
        groupby_col = self._target_col
        target_groupby_col = f"target_{groupby_col}"

        target_users = rank.loc[rank[groupby_col] == target_index, "user"].unique()
        filtered_rank = rank[rank.user.isin(target_users)]

        target_book_rank = filtered_rank[filtered_rank[groupby_col] == target_index]

        target_book_rank = target_book_rank.rename(columns={"rating": "target_rating", groupby_col: target_groupby_col})
        other_books_rank = filtered_rank[filtered_rank[groupby_col] != target_index]
        rank_target_other_book = pd.merge(other_books_rank, target_book_rank, on=["user"])
        # Calc mean, std for books.
        books_stats = (rank_target_other_book.groupby([groupby_col, target_groupby_col])
                       .agg({"rating": ["mean", "std"], "target_rating": ["mean", "std"]}).reset_index())
        books_stats.columns = [groupby_col, target_groupby_col, "mean", "std", "target_mean", "target_std"]
        # std == 0 <=> less than one rank value. Can't recommend in that case
        books_stats = books_stats[(books_stats['std'] > 0) & (books_stats['target_std'] > 0)]
        if len(books_stats) == 0:
            return books_stats
        books_stats = pd.merge(rank_target_other_book, books_stats, on=[groupby_col, target_groupby_col])
        # Added temp col for corr computations
        books_stats["almost_corr"] = (
                ((books_stats["rating"] - books_stats["mean"]) * (
                            books_stats["target_rating"] - books_stats["target_mean"]))
                / books_stats["std"] / books_stats["target_std"]
        )
        books_corr = books_stats.groupby([groupby_col, target_groupby_col]).agg({"almost_corr": "sum", "user": "count"})
        books_corr = books_corr.reset_index()
        books_corr["corr"] = books_corr["almost_corr"] / (books_corr["user"] - 1)
        return books_corr

    def predict(self: "CorrRecommender", example: str, max_rec: int=10) -> typing.List[str]:
        if not self._fitted:
            raise NotFittedError("Recommender is not fitted")
        if example not in self._target_val_set:
            print(f"{example} is unknown. Recommending top")
            return super(CorrRecommender, self).predict(example, max_rec)
        else:
            corr = self._calc_corr_for_df(example)
            if len(corr) == 0:
                print(f"Not enough data for recs. Recommending Popular")
                return super(CorrRecommender, self).predict(example, max_rec)
            max_corr = corr.sort_values("corr", ascending=False).iloc[:max_rec]
            recs = max_corr[self._target_col].to_list()
            cleaned_recs = [rec for rec in recs if rec != example]
            return cleaned_recs

    def __call__(self: "CorrRecommender", example: str, max_rec: int=10):
        return self.predict(example, max_rec)


class BookCorrRecommender(CorrRecommender):
    def __init__(self: "BookCorrRecommender"):
        super(BookCorrRecommender, self).__init__(target_col="title", rank_col="rating")


class AuthorCorrRecommender(CorrRecommender):
    def __init__(self: "AuthorCorrRecommender"):
        super(AuthorCorrRecommender, self).__init__(target_col="author", rank_col="rating")
