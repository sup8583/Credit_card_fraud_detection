from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html', error_message="")
    else:
        distance = request.form.get('distance_from_last_transaction')
        purchase_price = request.form.get('ratio_to_median_purchase_price')
        retailer = request.form.get('repeat_retailer')
        chip = request.form.get('used_chip')
        pin_number = request.form.get('used_pin_number')
        order = request.form.get('online_order')

        if any(val is None or val == "" for val in [distance, purchase_price, retailer, chip, pin_number, order]):
            return render_template('form.html', error_message="Please fill out all fields")

        try:
            distance = float(distance)
            purchase_price = float(purchase_price)
            retailer = float(retailer)
            chip = float(chip)
            pin_number = float(pin_number)
            order = float(order)
        except ValueError:
            return render_template('form.html', error_message="Invalid input values")

        data = CustomData(
            distance=distance,
            purchase_price=purchase_price,
            retailer=retailer,
            chip=chip,
            pin_number=pin_number,
            order=order
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)

if __name__ == '__main__':
    app.run(debug=True)
