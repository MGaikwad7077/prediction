from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor
import shap

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    shap_result = None

    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            return "Upload file"

        df = pd.read_excel(file)

        # ADD TIME FEATURE
        df['time'] = range(len(df))

        # FEATURES
        X = df[['time', 'likes', 'comments', 'subscribers']].values
        y = df['views'].values

        # TRAIN MODEL
        model = XGBRegressor(n_estimators=200, max_depth=4)
        model.fit(X, y)

        # 🔥 TRUE FUTURE PREDICTION (NO FAKE GROWTH)
        last_row = df.iloc[-1]

        next_time = last_row['time'] + 1

        future_input = [[
            next_time,
            last_row['likes'],
            last_row['comments'],
            last_row['subscribers']
        ]]

        predicted_views = model.predict(future_input)[0]

        # 🔥 ALSO PREDICT OTHER FEATURES (SEPARATE MODELS)

        model_likes = XGBRegressor()
        model_comments = XGBRegressor()
        model_subs = XGBRegressor()

        model_likes.fit(X, df['likes'].values)
        model_comments.fit(X, df['comments'].values)
        model_subs.fit(X, df['subscribers'].values)

        pred_likes = model_likes.predict(future_input)[0]
        pred_comments = model_comments.predict(future_input)[0]
        pred_subs = model_subs.predict(future_input)[0]

        predictions = {
            "views": round(predicted_views, 2),
            "likes": round(pred_likes, 2),
            "comments": round(pred_comments, 2),
            "subscribers": round(pred_subs, 2)
        }

        # 🔥 SHAP (REAL EXPLANATION)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        last_shap = shap_values[-1].values
        features = ['time', 'likes', 'comments', 'subscribers']

        total = sum(abs(v) for v in last_shap)

        shap_result = {
            features[i]: round((last_shap[i] / total) * 100, 2)
            for i in range(len(features))
        }

    return render_template(
        'index.html',
        predictions=predictions,
        shap_result=shap_result
    )

if __name__ == '__main__':
    import os
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)