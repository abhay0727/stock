import traceback

@app.route('/stock-analysis', methods=['POST'])
def stock_analysis():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        df = pd.read_csv(file)
        result = analyze_stock(df)
        return jsonify({'result': result})

    except Exception as e:
        print("🔥 ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
