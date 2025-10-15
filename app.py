from flask import Flask, jsonify
from stock_replinishment_Analysis import analyze_stock

app = Flask(__name__)

@app.route('/stock-analysis')
def stock_analysis():
    result = analyze_stock()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)