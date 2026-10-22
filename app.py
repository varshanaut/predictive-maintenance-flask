import os
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, session
)

app = Flask(__name__)
app.secret_key = 'predictive-maintenance-secret-key'

# Load the trained model
MODEL_PATH = Path(__file__).parent / 'predictive_model.pkl'

with open(MODEL_PATH, 'rb') as f:
    artefacts = pickle.load(f)

model = artefacts['model']
scaler = artefacts['scaler']
encoder = artefacts['encoder']
numeric_features = artefacts['numeric_features']
feature_names = artefacts['feature_names']
best_model_name = artefacts.get('best_model_name', 'Random Forest')


def engineer_features(df):
    df = df.copy()
    df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df['Wear_rate'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1e-6)
    return df


def transform(df):
    df = engineer_features(df)

    X_num = df[numeric_features]
    X_num_scaled = pd.DataFrame(
        scaler.transform(X_num),
        columns=numeric_features,
        index=df.index
    )

    type_enc = encoder.transform(df[['Type']])
    type_cols = encoder.get_feature_names_out(['Type']).tolist()
    type_df = pd.DataFrame(type_enc, columns=type_cols, index=df.index)

    X = pd.concat([X_num_scaled, type_df], axis=1)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X


def risk_category(proba):
    if proba < 0.30:
        return 'Low'
    if proba < 0.60:
        return 'Medium'
    return 'High'


def risk_badge(risk):
    return {'Low': 'success', 'Medium': 'warning', 'High': 'danger'}.get(risk, 'secondary')


@app.route('/')
def index():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', best_model_name=best_model_name)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        data = {
            'Type': request.form.get('Type', 'M').strip().upper(),
            'Air temperature [K]': float(request.form['air_temp']),
            'Process temperature [K]': float(request.form['process_temp']),
            'Rotational speed [rpm]': float(request.form['rpm']),
            'Torque [Nm]': float(request.form['torque']),
            'Tool wear [min]': float(request.form['tool_wear']),
        }

        if data['Type'] not in ('L', 'M', 'H'):
            flash('Type must be L, M or H', 'danger')
            return redirect(url_for('predict'))

        df = pd.DataFrame([data])
        X = transform(df)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])
        risk = risk_category(proba)

        result = {
            'prediction': pred,
            'label': 'FAILURE LIKELY' if pred == 1 else 'No Failure',
            'probability': round(proba * 100, 2),
            'risk': risk,
            'badge': risk_badge(risk),
            'inputs': data
        }
        return render_template('result.html', result=result)

    except Exception as e:
        flash(f'Invalid input: {e}', 'danger')
        return redirect(url_for('predict'))


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'GET':
        return render_template('batch.html')

    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('batch'))

    f = request.files['file']
    if f.filename == '' or not f.filename.lower().endswith('.csv'):
        flash('Please upload a CSV file', 'danger')
        return redirect(url_for('batch'))

    try:
        df = pd.read_csv(f)
        required = [
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            flash(f'Missing columns: {", ".join(missing)}', 'danger')
            return redirect(url_for('batch'))

        X = transform(df[required])
        probas = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        out = df.copy()
        out['Failure_Prediction'] = preds
        out['Failure_Probability'] = np.round(probas, 4)
        out['Risk_Category'] = [risk_category(p) for p in probas]

        buf = io.BytesIO()
        out.to_csv(buf, index=False)
        buf.seek(0)
        session['batch_csv'] = buf.getvalue().decode('utf-8')

        summary = {
            'total': len(out),
            'failures': int(preds.sum()),
            'high_risk': int((out['Risk_Category'] == 'High').sum()),
            'medium_risk': int((out['Risk_Category'] == 'Medium').sum()),
            'low_risk': int((out['Risk_Category'] == 'Low').sum()),
        }
        preview = out.head(50).to_dict(orient='records')
        columns = list(out.columns)

        return render_template('batch_result.html',
                               summary=summary,
                               preview=preview,
                               columns=columns)
    except Exception as e:
        flash(f'Error: {e}', 'danger')
        return redirect(url_for('batch'))


@app.route('/download_batch')
def download_batch():
    csv_data = session.get('batch_csv')
    if not csv_data:
        flash('No results to download', 'warning')
        return redirect(url_for('batch'))
    buf = io.BytesIO(csv_data.encode('utf-8'))
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True,download_name='batch_predictions.csv')


@app.route('/about')
def about():
    return render_template('about.html', best_model_name=best_model_name)


if __name__ == '__main__':
    app.run(debug=True)