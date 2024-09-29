from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)


url = "https://raw.githubusercontent.com/Akshat81172/CarPricePrediction/refs/heads/main/Cleaned_Car_data.csv"
df = pd.read_csv(url)

# Load the model
model = pickle.load(open("carpredict.pkl", "rb"))


@app.route('/')
def index():
    Car_Name = sorted(df['name'].unique())
    Company = sorted(df['company'].unique())
    Year = sorted(df['year'].unique(), reverse=True)
    Fuel_Type = df['fuel_type'].unique()
    return render_template("index.html", Company=Company, Car_Name=Car_Name, Year=Year, Fuel_Type=Fuel_Type)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    company = request.form.get('company')
    name = request.form.get('name')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kms_driven'))
    fuel_type = request.form.get('fuel_type')

    # Prepare the input data for the model
    input_values = [[name, company, year, kms_driven, fuel_type]]
    input_data = pd.DataFrame(input_values, columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Convert predicted price to an integer (remove decimals)
    predicted_price = int(predicted_price)

    # Render the form again with the prediction
    Car_Name = sorted(df['name'].unique())
    Company = sorted(df['company'].unique())
    Year = sorted(df['year'].unique(), reverse=True)
    Fuel_Type = df['fuel_type'].unique()

    return render_template("index.html", Company=Company, Car_Name=Car_Name, Year=Year, Fuel_Type=Fuel_Type,
                           predicted_price=predicted_price)


if __name__ == "__main__":
    app.run(debug=True)





































































































































































































