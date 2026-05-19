import os
import numpy as np
import pandas as pd
import joblib
import json

rain_model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "rain_classifier.pkl"))
temp_model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "temp_regressor.pkl"))
scaler     = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "scaler.pkl"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "reg_columns.json"), "r") as f:
    reg_columns = json.load(f)

SCALE_COLS = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
              "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
              "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
              "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]

def get_ml_prediction(input_dict):
    cls_input = pd.DataFrame([input_dict])
    cls_input[SCALE_COLS] = scaler.transform(cls_input[SCALE_COLS])
    rain_prob = rain_model.predict_proba(cls_input)[0][1]
    rain_pred = 1 if rain_prob >= 0.61 else 0
    reg_input      = pd.DataFrame([input_dict])[reg_columns]
    reg_scale_cols = [col for col in SCALE_COLS if col in reg_columns]
    scaled_all     = scaler.transform(pd.DataFrame([input_dict])[SCALE_COLS])
    col_indices    = [SCALE_COLS.index(c) for c in reg_scale_cols]
    reg_input[reg_scale_cols] = scaled_all[:, col_indices]
    temp_pred_scaled = temp_model.predict(reg_input)[0]
    dummy = np.zeros((1, len(SCALE_COLS)))
    dummy[0][SCALE_COLS.index("MaxTemp")] = temp_pred_scaled
    temp_pred_actual = scaler.inverse_transform(dummy)[0][SCALE_COLS.index("MaxTemp")]
    return {
        "rain_prob": round(float(rain_prob) * 100, 2),
        "rain_pred": "Yes" if rain_pred == 1 else "No",
        "temp_pred": round(float(temp_pred_actual), 2)
    }

def get_recommendations(rain_prob, max_temp, wind_speed, humidity):
    recommendations = {}

    if rain_prob >= 61:
        recommendations["Agriculture"] = "Avoid spraying pesticides. Good day for irrigation planning."
    elif max_temp > 40:
        recommendations["Agriculture"] = "Water crops early morning. Risk of heat stress on plants."
    else:
        recommendations["Agriculture"] = "Safe for plowing, planting and field work."

    if wind_speed > 120:
        recommendations["Construction"] = "Halt all outdoor construction. Dangerous wind conditions."
    elif rain_prob >= 61:
        recommendations["Construction"] = "Avoid concrete pouring and roofing work today."
    elif max_temp > 40:
        recommendations["Construction"] = "Limit outdoor work to early hours. Provide shade and water."
    else:
        recommendations["Construction"] = "Safe for all outdoor construction activities."

    if rain_prob >= 61:
        recommendations["Transportation"] = "Allow extra travel time. Roads may be slippery."
    elif wind_speed > 100:
        recommendations["Transportation"] = "Caution for high-sided vehicles. Avoid open highways."
    else:
        recommendations["Transportation"] = "Normal travel conditions. No major disruptions expected."

    if rain_prob >= 61:
        recommendations["Tourism"] = "Plan indoor attractions. Carry rain gear if going outside."
    elif max_temp > 40:
        recommendations["Tourism"] = "Avoid midday outdoor tours. Risk of heat exhaustion."
    else:
        recommendations["Tourism"] = "Excellent day for outdoor sightseeing and activities."

    if humidity > 80 and rain_prob >= 61:
        recommendations["Energy"] = "Low solar output expected. Wind energy may increase."
    elif max_temp > 38:
        recommendations["Energy"] = "High electricity demand expected. Peak load alert."
    else:
        recommendations["Energy"] = "Stable energy conditions. Good solar generation expected."

    if rain_prob >= 61 or wind_speed > 100:
        recommendations["Logistics"] = "Expect delivery delays. Reschedule non-urgent shipments."
    else:
        recommendations["Logistics"] = "Clear conditions for deliveries and warehouse operations."

    return recommendations

def get_alert_level(rain_prob, max_temp, wind_speed, rainfall):
    alerts = []
    level  = "Low"

    if max_temp > 40:
        alerts.append("HEAT ALERT - Extreme temperature above 40C")
        level = "High"
    elif max_temp > 35:
        alerts.append("HEAT WARNING - Temperature above 35C")
        if level == "Low":
            level = "Medium"

    if rain_prob >= 80:
        alerts.append("FLOOD ALERT - Very high rain probability")
        level = "High"
    elif rain_prob >= 61:
        alerts.append("RAIN WARNING - Moderate to high rain probability")
        if level == "Low":
            level = "Medium"

    if wind_speed > 120:
        alerts.append("STORM ALERT - Wind speed above 120 km/h")
        level = "High"
    elif wind_speed > 90:
        alerts.append("WIND WARNING - Strong winds above 90 km/h")
        if level == "Low":
            level = "Medium"

    if rainfall > 80:
        alerts.append("HEAVY RAINFALL ALERT - Rainfall above 80mm")
        level = "High"

    if len(alerts) == 0:
        alerts.append("No active weather alerts")

    return level, alerts