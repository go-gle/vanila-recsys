from utils import read_data
from recommenders import BookCorrRecommender, AuthorCorrRecommender
import argparse


def run():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--author", help="put an author name to see related")
    group.add_argument("-b", "--book", help="put aa book name to see related")
    args = parser.parse_args()
    df = read_data()
    if args.author:
        rec = AuthorCorrRecommender()
        rec.fit(df[["user", "rating", "author"]])
        example = args.author
    elif args.book:
        rec = BookCorrRecommender()
        rec.fit(df[["user", "rating", "title"]])
        example = args.book
    else:
        raise NotImplementedError(f"Not implemented for this argument")
    recs = rec(example.lower())
    print("Recs are")
    print(recs)

if __name__ == "__main__":
    run()



