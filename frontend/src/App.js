import React, { useState } from "react";
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "No",
    tenure: 12,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "yes",
    DeviceProtection: "No",
    TechSupport: "No", 
    StreamingTV: "Yes", 
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check", 
    MonthlyCharges: 70, 
    TotalCharges: 840
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const {name, value } = e.target;
    setFormData({ ...formData, [name]: value});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(formData)
    });

    const data = await response.json();
    setResult(data);
  };


  return (
    <div className="container">
      <h1>Customer Churn Predictor</h1>

      <form onSubmit={handleSubmit}>

        <label>Tenure (months)</label>
        <input name="tenure" type="number" value={formData.tenure} onChange={handleChange} />

        <label>Monthly Charges</label>
        <input name="MonthlyCharges" type="number" value={formData.MonthlyCharges} onChange={handleChange} />

        <label>Total Charges</label>
        <input name="TotalCharges" type="number" value={formData.TotalCharges} onChange={handleChange} />

        <label>Contract Type</label>
        <select name="Contract" value={formData.Contract} onChange={handleChange}>
          <option value="Month-to-month">Month-to-month</option>
          <option value="One year">One year</option>
          <option value="Two year">Two year</option>
        </select>

        <label>Internet Service</label>
        <select name="InternetService" value={formData.InternetService} onChange={handleChange}>
          <option value="DSL">DSL</option>
          <option value="Fiber optic">Fiber optic</option>
          <option value="No">No</option>
        </select>

        <label>Payment Method</label>
        <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
          <option value="Electronic check">Electronic check</option>
          <option value="Mailed check">Mailed check</option>
          <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
          <option value="Credit card (automatic)">Credit card (automatic)</option>
        </select>

        <button type="submit">Predict Churn</button>
      </form>

      {result && (
        <div className={`result ${result.churn_prediction === 1 ? "high" : "low"}`}>
          <h3>
            Prediction: {result.churn_prediction === 1 ? "Likely to Churn" : "Likely to Stay"}
          </h3>
          <p>Churn Probability: {result.churn_probability}</p>
        </div>
      )}
    </div>
  );
}

export default App;