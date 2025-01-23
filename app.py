from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import sqlite3
import os

import re
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import spacy
from wtpsplit import SaT
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import contractions

vocab_size = 5000
embedding_dim = 16
max_len = 100
trunction_type='post'
padding_type='post'
oov_token = "<OOV>"
from nltk.corpus import stopwords

aspect_model_dir = "./aspect_classification_model2"
aspect_model = BertForSequenceClassification.from_pretrained(aspect_model_dir)
aspect_tokenizer = BertTokenizer.from_pretrained(aspect_model_dir)
# spacy.cli.download("en_core_web_sm")

sentiment_model_path = "sentiment_model.h5"
sentiment_model = load_model(sentiment_model_path)
import string

with open("tokenizer.pkl", "rb") as f:
    sentiment_tokenizer = pickle.load(f)

def preprocessing(review):
    review = review.lower()
    review = review.translate(str.maketrans('', '', string.punctuation))


    conjunctions = r'\b(and|but|or|so|because|however|therefore|although|though|yet)\b'

    result = re.sub(conjunctions, '', review, flags=re.IGNORECASE)

    result = re.sub(r'\s+', ' ', result).strip()

    
    lemmatizer = WordNetLemmatizer()
    review = " ".join(lemmatizer.lemmatize(word) for word in review.split())
    review = contractions.fix(review)
    review = " ".join(review.split())

    return review

# nlp = spacy.load("en_core_web_sm")
# print("SpaCy model loaded successfully!")


# def split_sentences_spacy(text):
#     doc = nlp(text)
#     sentences = [sent.text.strip() for sent in doc.sents]
#     return sentences


def split_sentences_sat(text):
    sat_sm = SaT("sat-12l-sm")
    sat_sm.half().to("cuda")
    return sat_sm.split(text)

def split_into_sentences(review):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', review)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def classify_aspect(sentence):
    inputs = aspect_tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = aspect_model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    aspect_map = {0: "Fit", 1: "Fabric", 2: "Color", 3: "Design", 4: "Durability", 5: "Price", 6: "Comfort", 7: "Category", 8: "None of these"}
    return aspect_map[predicted_label]

def analyze_sentiment(sentence):
    test_sequence = sentiment_tokenizer.texts_to_sequences([sentence])
    test_padded = pad_sequences(test_sequence, maxlen=max_len, padding=padding_type, truncating=trunction_type)

    prediction = sentiment_model.predict(test_padded)
    return "Positive" if prediction[0][0] > 0.5 else "Negative"
    


app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATABASE = 'absa.db'

def initialize_database():
     with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                fit_preference REAL DEFAULT 0,
                fabric_preference REAL DEFAULT 0,
                color_preference REAL DEFAULT 0,
                design_preference REAL DEFAULT 0,
                durability_preference REAL DEFAULT 0,
                price_preference REAL DEFAULT 0,
                comfort_preference REAL DEFAULT 0,
                category_preference REAL DEFAULT 0
            )
        ''')

        cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', 
                            ('admin', 'admin', 'admin'))

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                color TEXT NOT NULL,
                size TEXT NOT NULL,
                image_path TEXT,
                fit_score REAL DEFAULT 0,
                fabric_score REAL DEFAULT 0,
                color_score REAL DEFAULT 0,
                design_score REAL DEFAULT 0,
                durability_score REAL DEFAULT 0,
                price_score REAL DEFAULT 0,
                comfort_score REAL DEFAULT 0,
                category_score REAL DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')

        conn.commit()

initialize_database()

def analyze_review_and_update_db(review, product_id):
    
    print(review)
    sentences = split_sentences_sat(review)
    print(sentences)
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentences.append(preprocessing(sentence))
    print(cleaned_sentences)

    aspect_scores = {aspect: None for aspect in [
        "Fit", "Fabric", "Color", "Design", "Durability", "Price", "Comfort", "Category"
    ]}

    for sentence in sentences:
        aspect = classify_aspect(sentence)
        if aspect != "None of these" and aspect_scores[aspect] is None:
            sentiment = analyze_sentiment(sentence)
            score = 1 if sentiment == "Positive" else -1
            aspect_scores[aspect] = score
        
        print(aspect, sentence, sentiment)

    averaged_scores = {key: score if score is not None else 0 for key, score in aspect_scores.items()}

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE products 
            SET fit_score = fit_score + ?,
                fabric_score = fabric_score + ?,
                color_score = color_score + ?,
                design_score = design_score + ?,
                durability_score = durability_score + ?,
                price_score = price_score + ?,
                comfort_score = comfort_score + ?,
                category_score = category_score + ?
            WHERE id = ?
        ''', (
            averaged_scores["Fit"],
            averaged_scores["Fabric"],
            averaged_scores["Color"],
            averaged_scores["Design"],
            averaged_scores["Durability"],
            averaged_scores["Price"],
            averaged_scores["Comfort"],
            averaged_scores["Category"],
            product_id
        ))
        conn.commit()

    return averaged_scores

@app.route('/static/uploads/<string:filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

@app.route('/product/<int:product_id>')
def product_details(product_id):

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        product = cursor.fetchone()

    if not product:
        return "Product not found", 404

    return render_template('review.html', product={
        'id': product[0],
        'name': product[1],
        'category': product[2],
        'price': product[3],
        'color': product[4],
        'size': product[5],
        'image': product[6]
    })

@app.route('/product/<int:product_id>/reviews', methods=['GET', 'POST'])
def product_reviews(product_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        if request.method == 'GET':
            # Fetch reviews for the specific product
            cursor.execute('SELECT username, content, date FROM reviews WHERE product_id = ?', (product_id,))
            reviews = cursor.fetchall()
            
            # Convert to list of dictionaries
            reviews_list = [
                {
                    'username': review[0], 
                    'content': review[1]
                } 
                for review in reviews
            ]
            
            return jsonify(reviews_list)
        
        elif request.method == 'POST':
            username = session.get('username', 'Anonymous')
            content = request.json.get('content')
            if not content:
                return jsonify({'status': 'error', 'message': 'Review cannot be empty'}), 400
    
            aspect_scores = analyze_review_and_update_db(content, product_id)
    
            # Update user preferences
            cursor.execute(f'''
                UPDATE users
                SET {', '.join(f"{aspect.lower()}_preference = {aspect.lower()}_preference + ?" for aspect in aspect_scores.keys())}
                WHERE username = ?
            ''', [*[abs(score) for score in aspect_scores.values()], username])
            conn.commit()

            # Insert review
            cursor.execute('INSERT INTO reviews (product_id, username, content) VALUES (?, ?, ?)',
                           (product_id, username, content))
            conn.commit()
            
            return jsonify({'status': 'success', 'message': 'Review added and analyzed successfully'})


def recommend_products(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        # Fetch user preferences
        cursor.execute('''
            SELECT fit_preference, fabric_preference, color_preference, design_preference, 
                   durability_preference, price_preference, comfort_preference, category_preference
            FROM users WHERE id = ?
        ''', (user_id,))
        user_preferences = cursor.fetchone()
        
        # Fetch products and calculate similarity scores
        cursor.execute('''
            SELECT id, name, category, price, color, size, image_path, 
                   fit_score, fabric_score, color_score, design_score, 
                   durability_score, price_score, comfort_score, category_score
            FROM products
        ''')
        products = cursor.fetchall()
        
        recommendations = []
        for product in products:
            product_id, name, category, price, color, size, image_path, *scores = product
            similarity_score = sum(
                up * ps for up, ps in zip(user_preferences, scores)
            )  # Dot product of user preferences and product scores
            recommendations.append({
                "id": product_id,
                "name": name,
                "category": category,
                "price": price,
                "color": color,
                "size": size,
                "image": image_path,
                "score": similarity_score,
            })
        
        # Sort by similarity score in descending order
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        print(recommendations)

        return recommendations[:5]  # Return top 5 recommendations

@app.route('/recommend')
def recommend():
    # Ensure the user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    recommendations = recommend_products(user_id)
    return render_template('recommended.html', recommendations=recommendations)


@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Check if passwords match
        if password != confirm_password:
            return render_template('signup.html', message='Passwords do not match!')

        # Save to database
        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                print(username)
                print(password)
                cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, 'user'))
                conn.commit()
            return redirect(url_for('login', message='Account created successfully!'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', message='Username already exists!')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate credentials
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
            user = cursor.fetchone()

        if user:
            session['username'] = username
            session['user_id'] = user[0]
            session['role'] = user[3]  # Assuming role is the 4th column in the table (index 3)

            # Redirect based on role
            if user[3] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            return render_template('login.html', message='Invalid username or password')

    # Render the login page
    return render_template('login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'role' in session and session['role'] == 'admin':
        return render_template('create.html')
    return redirect(url_for('login'))

@app.route('/user/dashboard')
def user_dashboard():
    if 'role' in session and session['role'] == 'user':
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, category, price, color, size, image_path FROM products")
            rows = cursor.fetchall()
            products = [
                {
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'price': row[3],
                    'color': row[4],
                    'size': row[5],
                    'image': row[6],  
                }
                for row in rows
            ]
        return render_template('marketplace.html', products=products)
    return redirect(url_for('login'))

@app.route('/')
def landing_page():
    return render_template('landingpage.html')

@app.route('/create_product', methods=['POST'])
def create_product():
    if 'role' in session and session['role'] == 'admin':
        name = request.form.get('name')
        category = request.form.get('category')
        price = request.form.get('price')
        color = request.form.get('color')
        size = request.form.get('size')
        image = request.files.get('image')

        image_path = None
        if image:
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            image_path = upload_folder + '/' + image.filename
            image.save(image_path)

        # Save product data to the database
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO products (name, category, price, color, size, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, category, price, color, size, image.filename))
            conn.commit()

        return jsonify({'status': 'success', 'message': 'Product created successfully!'}), 200

    return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    

@app.route('/logout')
def logout():
    session.pop('username', None)  # Clear the session
    print('logingout')
    return redirect(url_for('login'))

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True)
