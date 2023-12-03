import pandas as pd
import numpy as np


DATA_PATH = "../../../"

def prep_data(books: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher']
    books.loc[books.ISBN == '0789466953', 'yearOfPublication'] = 2000
    books.loc[books.ISBN == '0789466953', 'bookAuthor'] = "James Buckley"
    books.loc[books.ISBN == '0789466953', 'publisher'] = "DK Publishing Inc"
    books.loc[
        books.ISBN == '0789466953', 'bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

    books.loc[books.ISBN == '078946697X', 'yearOfPublication'] = 2000
    books.loc[books.ISBN == '078946697X', 'bookAuthor'] = "Michael Teitelbaum"
    books.loc[books.ISBN == '078946697X', 'publisher'] = "DK Publishing Inc"
    books.loc[
        books.ISBN == '078946697X', 'bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

    books.loc[books.ISBN == '2070426769', 'yearOfPublication'] = 2003
    books.loc[books.ISBN == '2070426769', 'bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
    books.loc[books.ISBN == '2070426769', 'publisher'] = "Gallimard"
    books.loc[books.ISBN == '2070426769', 'bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"

    books.loc[(books.ISBN == '193169656X'), 'publisher'] = 'other'
    books.loc[(books.ISBN == '1931696993'), 'publisher'] = 'other'

    books.yearOfPublication = pd.to_numeric(books.yearOfPublication, errors='coerce')

    books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
    books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
    books.yearOfPublication = books.yearOfPublication.astype(np.int32)

    ratings = ratings[ratings["ISBN"].isin(set(books['ISBN'].unique()))]
    books = books.rename(columns={"ISBN": "isbn", "bookTitle": "title", "bookAuthor": "author"})
    books.title = books.title.str.lower()
    books.author = books.author.str.lower()
    ratings = ratings.rename(columns={"ISBN": "isbn", "User-ID": "user", "Book-Rating": "rating"})
    return pd.merge(books, ratings, on="isbn")


def read_data() -> pd.DataFrame:
    books = pd.read_csv(F"{DATA_PATH}Books.csv")
    ratings = pd.read_csv(f"{DATA_PATH}Ratings.csv")
    return prep_data(books, ratings)
