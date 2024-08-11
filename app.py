import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Đọc dữ liệu và loại bỏ các giá trị NaN
data = pd.read_csv('models/clean_data.csv').dropna(subset=['Name', 'Category'])

# Xử lý dữ liệu
X = data['Name']
y = data['Category']

# Chuyển đổi dữ liệu văn bản
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Tạo và huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_transformed, y)

# Lưu mô hình
joblib.dump((model, vectorizer), 'models/random_forest_model.pkl')

# Tải mô hình
model, vectorizer = joblib.load('models/random_forest_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    product_name = request.form['product_name']

    # Kiểm tra và xử lý giá trị NaN
    if not product_name.strip():
        return jsonify({'error': 'Please enter a valid product name.'})

    # Chuyển đổi tên sản phẩm
    product_name_transformed = vectorizer.transform([product_name])

    # Dự đoán
    prediction = model.predict(product_name_transformed)

    # Lấy sản phẩm tương tự từ dữ liệu gốc
    similar_products = data[data['Category'] == prediction[0]]

    # Lấy 5 sản phẩm tương tự nhất
    recommendations = similar_products.head(10).to_dict(orient='records')

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
