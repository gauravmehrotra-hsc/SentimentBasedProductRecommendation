import pickle
import pandas as pd
import numpy as np

# Load the necessary models and data
user_final_rating = pickle.load(open("pickle/user_final_rating.pkl", "rb"))
df_final = pickle.load(open("pickle/cleaned-data.pkl", "rb"))
tfidf = pickle.load(open("pickle/tfidf-vectorizer.pkl", "rb"))
xgb = pickle.load(open("pickle/sentiment-classification-xg-boost-model.pkl", "rb"))


def product_recommendations_user(user):
    if (user in user_final_rating.index):
        # Get top 20 recommended product IDs from the best recommendation model
        top20_recommended_product_ids = user_final_rating.loc[user].sort_values(ascending=False)[0:20].index.tolist()

        # Get relevant product details from df_final for these recommended products
        df_top20_products = df_final[df_final['id'].isin(top20_recommended_product_ids)].drop_duplicates(subset=['id', 'cleaned_review']).copy() # Use .copy() to avoid SettingWithCopyWarning

        if df_top20_products.empty:
            print(f"No product details found for recommended IDs for user {user}.")
            return pd.DataFrame() # Return empty DataFrame if no products found

        # Ensure review_length is present and up-to-date for the current subset of products
        df_top20_products['review_length'] = df_top20_products['cleaned_review'].apply(len)

        # Transform text features
        X_tfidf = tfidf_vectorizer.transform(df_top20_products['cleaned_review'].values.astype(str))

        # Scale numeric feature (review_length)
        local_review_length_scaler = MinMaxScaler()
        # Fit on original X_train review_length, then transform current subset
        # Ensure X_train is available from the global scope or loaded.
        local_review_length_scaler.fit(X_train[['review_length']])
        X_num = local_review_length_scaler.transform(df_top20_products[['review_length']])

        # Combine features
        X_combined = hstack((X_tfidf, X_num))

        # Print shapes for debugging and verification
        print(f"Shape of X_tfidf for recommendations: {X_tfidf.shape}")
        # Use get_feature_names_out() to accurately show the number of features the vectorizer was fitted with
        print(f"Number of features in TF-IDF vocabulary: {len(tfidf_vectorizer.get_feature_names_out())}")
        print(f"Shape of X_num for recommendations: {X_num.shape}")
        print(f"Shape of X_combined before prediction: {X_combined.shape}")
        expected_features = model_xgb.n_features_in_ if hasattr(model_xgb, 'n_features_in_') else 'unknown'
        print(f"Model expects {expected_features} features.")

        # Predict sentiment
        df_top20_products['predicted_sentiment'] = model_xgb.predict(X_combined)

        # Calculate total reviews and positive reviews per product name in one go
        sentiment_summary_df = df_top20_products.groupby('name').agg(
            total_review_count=('predicted_sentiment', 'size'),
            pos_review_count=('predicted_sentiment', lambda x: (x == 1).sum())
        ).reset_index()

        # Calculate positive sentiment percentage
        sentiment_summary_df['pos_sentiment_percentage'] = np.round(
            (sentiment_summary_df['pos_review_count'] / sentiment_summary_df['total_review_count']) * 100, 2
        )

        # Ensure 'pos_sentiment_percentage' is numeric before sorting to avoid TypeError
        sentiment_summary_df['pos_sentiment_percentage'] = pd.to_numeric(sentiment_summary_df['pos_sentiment_percentage'], errors='coerce').fillna(0)

        # Return top 5 recommended products sorted by positive sentiment percentage
        result = sentiment_summary_df.sort_values(by='pos_sentiment_percentage', ascending=False)[:5][['name', 'pos_sentiment_percentage']]
        return result
    else:
        print(f"User name {user} doesn't exist")
        return pd.DataFrame() # Return empty DataFrame for non-existent user



#def product_recommendations_user(user_name):
#    """Returns top 5 recommended products for a given user along with sentiment scores"""
#    if user_name not in user_final_rating.index:
#        return f"The user '{user_name}' does not exist. Please provide a valid user name."

#    top20_recommended_products = list(user_final_rating.loc[user_name].sort_values(ascending=False)[:20].index)

#    df_top20_products = df_final[df_final.name.isin(top20_recommended_products)].drop_duplicates(subset=['cleaned_review'])

#    if df_top20_products.empty:
#        return "No recommendations found for this user."

    # Transform text using TF-IDF
#    X = tfidf.transform(df_top20_products['cleaned_review'])
#    X_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

    # Include numerical features
#    X_num = df_top20_products[['review_length']].reset_index(drop=True)
#    X_df = X_df.reset_index(drop=True)

#    df_top_20_products_final_features = pd.concat([X_df, X_num], axis=1)

    # Predict sentiment
#    df_top20_products['predicted_sentiment'] = xgb.predict(df_top_20_products_final_features)

    # Process sentiment results
#    df_top20_products['positive_sentiment'] = df_top20_products['predicted_sentiment'].apply(lambda x: 1 if x == 1 else 0)

#    pred_df = df_top20_products.groupby(by='name').sum()
#    pred_df = pred_df.rename(columns={'positive_sentiment': 'pos_sent_count'})

#    pred_df['total_sent_count'] = df_top20_products.groupby(by='name')['predicted_sentiment'].count()
#    pred_df['pos_sent_percentage'] = np.round(pred_df['pos_sent_count'] / pred_df['total_sent_count'] * 100, 2)

#    pred_df = pred_df.reset_index()

    # Return top 5 recommended products
#    return pred_df.sort_values(by="pos_sent_percentage", ascending=False)[:5][["name", "pos_sent_percentage"]]
