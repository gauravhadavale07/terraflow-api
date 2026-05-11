# TerraFlow BioAI — Prediction API

## Files needed in your GitHub repo
```
terraflow_rf_model.joblib   ← trained model
app.py                      ← Flask API
requirements.txt            ← dependencies
Procfile                    ← Render start command
```

---

## Deploy on Render (Free)

### Step 1 — Create GitHub repo
1. Go to github.com → New repository → name it `terraflow-api`
2. Upload all 4 files above

### Step 2 — Deploy on Render
1. Go to render.com → Sign up (free)
2. Click **New → Web Service**
3. Connect your GitHub account → select `terraflow-api` repo
4. Fill in settings:
   - **Name:** terraflow-api
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Click **Deploy** → wait ~2 minutes
6. Your live URL: `https://terraflow-api.onrender.com`

---

## API Usage

### Health Check
```
GET https://terraflow-api.onrender.com/health
```

### Predict
```
POST https://terraflow-api.onrender.com/predict
Content-Type: application/json

{
  "moisture": 22,
  "soil_temp": 32,
  "tds": 400,
  "ambient_temp": 35,
  "humidity": 35,
  "pressure": 1010,
  "lux": 60000
}
```

### Response
```json
{
  "prediction": "IRRIGATE",
  "confidence": 0.9733,
  "label": 1,
  "indices": {
    "wsi": 0.748,
    "dri": 0.087
  }
}
```

---

## Call from Node.js backend

```javascript
const axios = require('axios');

async function getIrrigationPrediction(sensorData) {
  const response = await axios.post(
    'https://terraflow-api.onrender.com/predict',
    {
      moisture:     sensorData.moisture,
      soil_temp:    sensorData.soil_temp,
      tds:          sensorData.tds,
      ambient_temp: sensorData.amb_temp,
      humidity:     sensorData.humidity,
      pressure:     sensorData.pressure,
      lux:          sensorData.lux
    }
  );
  return response.data; // { prediction, confidence, label, indices }
}
```

---

## ⚠️ Render Free Tier Note
Free tier **spins down after 15 min of inactivity**.
First request after sleep takes ~30 seconds — subsequent requests are fast.

To keep it warm, ping `/health` every 10 minutes using:
- cron-job.org (free)
- UptimeRobot (free)
