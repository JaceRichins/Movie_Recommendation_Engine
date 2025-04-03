import pandas as pd
import numpy as np
import random
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Suppress DataConversionWarning
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics.pairwise')


class MovieRecommender:
    def __init__(self, movies_file="movies.csv", movies_description_file="movies_description.csv"):
        self.movies_description_df = pd.read_csv(movies_description_file)
        self.movies_df = pd.read_csv(movies_file)
        # Drop the title to avoid dupicate
        self.movies_description_df.drop(columns=["original_title"], inplace=True)
        self.movies_description_df['imdb_id'] = self.movies_description_df['imdb_id'].str.replace('tt0', '')

        self.movies_df['imdb_id'] = self.movies_df['imdb_id'].astype(str)
        self.movies_description_df['imdb_id'] = self.movies_description_df['imdb_id'].astype(str)

        self.movies_df = pd.merge(self.movies_df, self.movies_description_df, on="imdb_id", how="inner")

        #############################################
        # print(self.movies_df[self.movies_df['overview'].isna()]['imdb_id'])
        # print(self.movies_df[['title','overview']].head(35))
        # print(self.movies_df.columns)

        #####################

        # Extract year from title and create a new column
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype('float')
        # Create clean title without year
        self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)


        self.movies_df['genre_list'] = self.movies_df['genres'].str.split('|')


        self.mlb = MultiLabelBinarizer()
        self.genre_matrix = self.mlb.fit_transform(self.movies_df['genre_list'])


        self.tfidf = TfidfVectorizer(stop_words='english')
        self.title_matrix = self.tfidf.fit_transform(self.movies_df['clean_title'])



        self.tfidf_description = TfidfVectorizer(stop_words='english')
        self.description_matrix = self.tfidf_description.fit_transform(self.movies_df['overview'].fillna(''))


        self.genre_model = self._build_genre_knn_model()
        self.title_model = self._build_title_knn_model()
        self.description_model = self._build_description_knn_model()


        self.user_selections = []
        self.current_recommendations = self._get_random_movies(10)


    def _build_genre_knn_model(self, k=10):
        model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='jaccard')
        model.fit(self.genre_matrix.astype(bool))
        return model

    def _build_title_knn_model(self, k=10):
        model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine')
        model.fit(self.title_matrix)
        return model

    def _build_description_knn_model(self, k=10):
        model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine')
        model.fit(self.description_matrix)
        return model

    def _get_random_movies(self, n=10):
        return random.sample(list(self.movies_df.index), n)

    def _get_recommendations(self, k=10):
        """Get recommendations based on all user selections."""
        if not self.user_selections:
            return self._get_random_movies(k)

        all_genre_recommendations = []
        all_title_recommendations = []
        all_year_recommendations = []
        all_description_recommendations = []


        for selected_movie in self.user_selections:

            genre_features = self.genre_matrix[selected_movie].reshape(1, -1)
            _, genre_indices = self.genre_model.kneighbors(genre_features)
            all_genre_recommendations.extend(genre_indices[0])


            title_features = self.title_matrix[selected_movie]
            _, title_indices = self.title_model.kneighbors(title_features)
            all_title_recommendations.extend(title_indices[0])

            description_features = self.description_matrix[selected_movie]
            _, description_indices = self.description_model.kneighbors(description_features)
            all_description_recommendations.extend(title_indices[0])

            selected_year = self.movies_df.iloc[selected_movie]['year']
            year_mask = abs(self.movies_df['year'] - selected_year) <= 3

            sample_size = min(k // len(self.user_selections) + 1, year_mask.sum())
            all_year_recommendations.extend(self.movies_df[year_mask].sample(sample_size).index)


        all_recommendations = (all_genre_recommendations * 2) + (all_description_recommendations * 2)

        recommendation_counts = Counter(all_recommendations)

        for selected in self.user_selections:
            if selected in recommendation_counts:
                del recommendation_counts[selected]


        most_common = [movie_idx for movie_idx, _ in recommendation_counts.most_common(k)]


        if len(most_common) < k:
            random_indices = [idx for idx in self._get_random_movies(k)
                              if idx not in most_common and idx not in self.user_selections]
            most_common.extend(random_indices[:k - len(most_common)])

        return most_common[:k]

    def search_movies(self, query, search_type="title"):
        if search_type == "title":
            # Search in movie titles
            results = self.movies_df[self.movies_df['clean_title'].str.contains(query, case=False)].index.tolist()
        elif search_type == "year":
            # Try to convert query to year
            try:
                year = int(query)
                results = self.movies_df[self.movies_df['year'] == year].index.tolist()
            except (ValueError, TypeError):
                print(f"Invalid year: {query}")
                results = []
        elif search_type == "year_range":
            # Try to parse year range (format: start-end)
            try:
                start_year, end_year = map(int, query.split('-'))
                results = self.movies_df[(self.movies_df['year'] >= start_year) &
                                         (self.movies_df['year'] <= end_year)].index.tolist()
            except (ValueError, TypeError):
                print(f"Invalid year range: {query}. Use format 'start-end' (e.g., '1990-2000')")
                results = []
        else:
            print(f"Invalid search type: {search_type}")
            results = []


        self.current_recommendations = results[:10] if len(results) > 10 else results
        return self.current_recommendations

    def clear_search(self):
        self.current_recommendations = self._get_recommendations(10)
        return self.current_recommendations

    def select_movie(self, index):
        movie_idx = self.current_recommendations[index]
        self.user_selections.append(movie_idx)
        self.current_recommendations = self._get_recommendations(10)
        return self.current_recommendations

    def get_new_random_movies(self):
        self.current_recommendations = self._get_random_movies(10)
        return self.current_recommendations

    def get_movie_details(self, movie_idx):
        movie = self.movies_df.iloc[movie_idx]
        return {
            'id': movie_idx,
            'title': movie['title'],
            'year': movie['year'],
            'genres': movie['genres'],
            'clean_title': movie['clean_title']
        }

    def get_current_recommendations(self):
        return [self.get_movie_details(idx) for idx in self.current_recommendations]

    def get_user_selections(self):
        return [self.get_movie_details(idx) for idx in self.user_selections]


def main():
    try:
        recommender = MovieRecommender()
        recommendations = recommender.get_current_recommendations()

        running = True
        while running:
            print("\n=== MOVIE RECOMMENDATIONS ===")
            for i, movie in enumerate(recommendations):
                print(f"{i + 1}. {movie['title']} ({int(movie['year']) if not pd.isna(movie['year']) else 'N/A'}) - {movie['genres']}")

            # Display user selections
            if recommender.user_selections:
                print("\n=== YOUR SELECTED MOVIES ===")
                for i, movie in enumerate(recommender.get_user_selections()):
                    print(f"{i + 1}. {movie['title']}")


            print("\n=== MENU ===")
            print("Enter the number (1-10) to select a movie")
            print("Enter 'r' for new random movies")
            print("Enter 's' to search for movies (by title, year, or year range)")
            print("Enter 'c' to clear search results")
            print("Enter 'q' to quit")

            choice = input("\nYour choice: ").strip().lower()

            if choice == 'q':
                running = False
                print("Thank you for using the Movie Recommender!")

            elif choice == 'r':
                recommendations = recommender.get_new_random_movies()
                recommendations = recommender.get_current_recommendations()
                print("New random movies generated!")

            elif choice == 's':
                search_type = input("Search by title, year, or year range? (t/y/r): ").strip().lower()

                if search_type == 't':
                    query = input("Enter movie title: ").strip()
                    recommender.search_movies(query, "title")
                elif search_type == 'y':
                    query = input("Enter year: ").strip()
                    recommender.search_movies(query, "year")
                elif search_type == 'r':
                    query = input("Enter year range (e.g., 1990-2000): ").strip()
                    recommender.search_movies(query, "year_range")
                else:
                    print("Invalid search type. Please try again.")
                    continue

                recommendations = recommender.get_current_recommendations()
                if not recommendations:
                    print("No movies found! Showing recommendations instead.")
                    recommender.clear_search()
                    recommendations = recommender.get_current_recommendations()

            elif choice == 'c':
                recommender.clear_search()
                recommendations = recommender.get_current_recommendations()
                print("Search cleared. Showing recommendations.")

            elif choice.isdigit() and 1 <= int(choice) <= len(recommendations):
                index = int(choice) - 1
                recommender.select_movie(index)
                print(f"Added '{recommendations[index]['title']}' to your selections!")
                recommendations = recommender.get_current_recommendations()

            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()