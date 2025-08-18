# Amazon-Product-Recommendation-System
Built a recommendation system that recommends products to Amazon customers based on their review history using Sci-kit

## Note
The training, test dataset and the saved pickle files for the model reside on a GCP server and are too large to be uploaded onto GitHub (> 200 MB).

# Notebook Order

## 1. `data_loading.ipynb`
- Loads compressed **CSV/JSONL** files from Amazon reviews dataset  
- Processes user interactions (**ratings, timestamps, purchase history**)  
- Extracts product metadata (**titles, prices, ratings, store details**)  
- Saves preprocessed datasets in **Parquet format** for efficient storage  

---

## 2. `exploratory_data_analysis.ipynb`
- Analyzes **6M+ users** and **2.9M products** across 14 categories  
- Plots **user behavior patterns** and **product characteristics**  

---

## 3. `popularity_model.ipynb`
- Implements a **popularity-based recommender** using review counts and ratings  
- Serves as a **baseline** with a **0.38% hit rate** across categories  

---

## 4. `matrix_factorization_model.ipynb`
- Implements **Alternating Least Squares (ALS)** collaborative filtering via the Implicit library  
- Handles sparsity using **confidence weighting**  
- Evaluates performance on **known users/products**  
- Achieves **0.39% Precision@10** with **0.502 AUC**  

---

## 5. `hybrid_matrix_factorization_model.ipynb`
- Integrates **collaborative filtering** with **content-based features**  
- Incorporates product metadata (**categories, price bins, rating bins**)  
- Uses **LightFM** for hybrid matrix factorization  
- Best-performing model: **0.41% Precision@10** with **0.867 AUC**  
