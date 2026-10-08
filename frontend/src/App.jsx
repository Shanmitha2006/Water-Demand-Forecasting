import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const navItems = [
  "Overview",
  "Dataset",
  "Data Analysis",
  "Forecasting",
  "Model Comparison",
  "Power BI Export",
  "About Project",
];

const formatNumber = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
};

function App() {
  const [activeTab, setActiveTab] = useState("Overview");
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [edaData, setEdaData] = useState(null);
  const [forecastResults, setForecastResults] = useState([]);
  const [predictionResults, setPredictionResults] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [targetOptions, setTargetOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    fetchDatasetSummary();
  }, []);

  const fetchDatasetSummary = async () => {
    setLoading(true);
    setErrorMessage("");
    try {
      const response = await axios.get(`${API_BASE}/api/dataset-summary`);
      const summary = response.data;
      setErrorMessage("");
      setDatasetSummary(summary);
      setTargetOptions(summary.target_candidates || []);
      setSelectedTarget(summary.target_candidates?.[0] || "");

      const edaResponse = await axios.get(`${API_BASE}/api/eda`);
      setEdaData(edaResponse.data.eda);
      const [comparisonResponse, predictionsResponse, forecastResponse] =
        await Promise.all([
          axios.get(`${API_BASE}/api/model-comparison`),
          axios.get(`${API_BASE}/api/predictions`),
          axios.get(`${API_BASE}/api/forecast`),
        ]);
      setModelComparison(comparisonResponse.data.results || []);
      setPredictionResults(predictionsResponse.data.predictions || []);
      setForecastResults(forecastResponse.data.forecast || []);
    } catch (error) {
      console.error("Failed to fetch summary", error);
      setErrorMessage(
        error.response
          ? "Unable to load dataset"
          : "Unable to connect to backend",
      );
    } finally {
      setLoading(false);
    }
  };

  const trainModels = async () => {
    const payload = {
      target_column: selectedTarget || undefined,
      forecast_horizon: 7,
    };
    setLoading(true);
    setErrorMessage("");
    try {
      const response = await axios.post(`${API_BASE}/api/train`, payload);
      setModelComparison(response.data.results || []);
      setPredictionResults(response.data.predictions || []);
      setSelectedTarget(response.data.target_column || selectedTarget);
    } catch (error) {
      console.error("Failed to train models", error);
      setErrorMessage("Training failed");
    } finally {
      setLoading(false);
    }
  };
  const generateForecast = async () => {
    setLoading(true);
    setErrorMessage("");
    try {
      const response = await axios.post(`${API_BASE}/api/forecast`, {
        target_column: selectedTarget || undefined,
        forecast_horizon: 7,
      });
      setForecastResults(response.data.forecast || []);
    } catch (error) {
      console.error("Failed to forecast", error);
      setErrorMessage(
        error.response?.data?.detail || "Forecast generation failed.",
      );
    } finally {
      setLoading(false);
    }
  };

  const kpis = useMemo(() => {
    if (!datasetSummary) return [];
    const stats = datasetSummary.statistics || {};
    return [
      {
        label: "Total Records",
        value: formatNumber(datasetSummary.rows),
      },
      { label: "Countries", value: formatNumber(stats.country_count) },
      {
        label: "Year Range",
        value: stats.year_min ? `${stats.year_min}-${stats.year_max}` : "—",
      },
      { label: "Average Demand", value: formatNumber(stats.average) },
      { label: "Maximum Demand", value: formatNumber(stats.maximum) },
      { label: "Minimum Demand", value: formatNumber(stats.minimum) },
      { label: "Best Model", value: modelComparison[0]?.model || "—" },
      {
        label: "Forecast Accuracy",
        value: modelComparison[0]?.r2
          ? `${modelComparison[0].r2.toFixed(2)}`
          : "—",
      },
    ];
  }, [datasetSummary, modelComparison]);

  const renderOverview = () => (
    <div className="page">
      <section className="hero panel">
        <div className="hero-copy">
          <p className="eyebrow">AI-powered analytics</p>
          <h1>AI-Driven Water Demand Forecasting</h1>
          <p className="subtitle">
            Predict future water demand using machine learning and transform
            consumption data into actionable insights.
          </p>
          <div className="cta-row">
            <button
              className="primary-btn"
              onClick={() => setActiveTab("Data Analysis")}
            >
              Explore Analytics
            </button>
            <button
              className="secondary-btn"
              onClick={() => setActiveTab("Forecasting")}
            >
              Run Forecast
            </button>
          </div>
        </div>
        <div className="hero-visual">
          <div className="wave wave-one"></div>
          <div className="wave wave-two"></div>
          <div className="glass-card metric-box">
            <span>Demand Index</span>
            <strong>{datasetSummary ? datasetSummary.rows : "—"}</strong>
            <small>records loaded</small>
          </div>
        </div>
      </section>

      <section className="kpi-grid">
        {kpis.map((item) => (
          <div className="panel kpi-card" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </section>

      <section className="grid-2">
        <div className="panel chart-panel">
          <h3>Water Demand Trend</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={edaData?.monthly_summary || []}>
                <defs>
                  <linearGradient id="waterFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#2ec4ff" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#2ec4ff" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date_label" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="demand"
                  fill="url(#waterFill)"
                  stroke="#15a7d9"
                  strokeWidth={3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel chart-panel">
          <h3>Model Comparison</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rmse" fill="#00b8d9" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );

  const renderDataset = () => (
    <div className="page">
      <div className="panel dataset-summary">
        <h3>Dataset Overview</h3>
        {datasetSummary ? (
          <div className="summary-grid">
            <div>
              <strong>Dataset name:</strong> {datasetSummary.dataset_name}
            </div>
            <div>
              <strong>Rows:</strong> {datasetSummary.rows}
            </div>
            <div>
              <strong>Columns:</strong> {datasetSummary.columns}
            </div>
            <div>
              <strong>Duplicate rows:</strong> {datasetSummary.duplicate_rows}
            </div>
            <div>
              <strong>Date columns:</strong>{" "}
              {datasetSummary.datetime_columns?.join(", ") || "None detected"}
            </div>
            <div>
              <strong>Potential targets:</strong>{" "}
              {datasetSummary.target_candidates?.join(", ") || "Not detected"}
            </div>
          </div>
        ) : null}
      </div>

      <div className="panel">
        <h3>Column Details</h3>
        {datasetSummary ? (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Type</th>
                  <th>Missing</th>
                </tr>
              </thead>
              <tbody>
                {datasetSummary.column_names?.map((col) => (
                  <tr key={col}>
                    <td>{col}</td>
                    <td>{datasetSummary.dtypes?.[col] || "unknown"}</td>
                    <td>{datasetSummary.missing_values?.[col] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </div>
  );

  const renderAnalysis = () => (
    <div className="page">
      <div className="panel filter-box">
        <label>
          Target variable
          <select
            value={selectedTarget}
            onChange={(e) => setSelectedTarget(e.target.value)}
          >
            <option value="">Auto-detect</option>
            {targetOptions.map((opt) => (
              <option value={opt} key={opt}>
                {opt}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="grid-2">
        <div className="panel chart-panel">
          <h3>Distribution</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={edaData?.monthly_summary || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date_label" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="demand" fill="#00b8d9" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel chart-panel">
          <h3>Monthly Demand</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={edaData?.monthly_summary || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date_label" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="demand"
                  stroke="#1aa7d8"
                  strokeWidth={3}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );

  const renderForecast = () => (
    <div className="page">
      <div className="panel filter-box">
        <label>
          Target Variable
          <select
            value={selectedTarget}
            onChange={(e) => setSelectedTarget(e.target.value)}
          >
            <option value="">Auto-detect</option>
            {targetOptions.map((opt) => (
              <option value={opt} key={opt}>
                {opt}
              </option>
            ))}
          </select>
        </label>
        <button className="primary-btn" onClick={trainModels}>
          Train Models
        </button>
        <button className="secondary-btn" onClick={generateForecast}>
          Generate Forecast
        </button>
      </div>

      <div className="grid-2">
        <div className="panel chart-panel">
          <h3>Actual vs Predicted</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={predictionResults}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Index" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="Predicted Demand"
                  stroke="#00b8d9"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="Actual Demand"
                  stroke="#2ecc71"
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel chart-panel">
          <h3>Forecast</h3>
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={forecastResults}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Date" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="Predicted Demand"
                  stroke="#2ecc71"
                  strokeWidth={3}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel chart-panel">
          <h3>Best Model Metrics</h3>
          {modelComparison[0] ? (
            <div className="metric-list">
              <div>
                <span>Model</span>
                <strong>{modelComparison[0].model}</strong>
              </div>
              <div>
                <span>MAE</span>
                <strong>{modelComparison[0].mae?.toFixed(4)}</strong>
              </div>
              <div>
                <span>RMSE</span>
                <strong>{modelComparison[0].rmse?.toFixed(4)}</strong>
              </div>
              <div>
                <span>R²</span>
                <strong>{modelComparison[0].r2?.toFixed(4)}</strong>
              </div>
              <div>
                <span>MAPE</span>
                <strong>{modelComparison[0].mape?.toFixed(4)}</strong>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );

  const renderComparison = () => (
    <div className="page">
      <div className="panel">
        <h3>Model Comparison</h3>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>MAPE</th>
              </tr>
            </thead>
            <tbody>
              {modelComparison.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.mae?.toFixed(4)}</td>
                  <td>{row.rmse?.toFixed(4)}</td>
                  <td>{row.r2?.toFixed(4)}</td>
                  <td>{row.mape?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderPowerbi = () => (
    <div className="page">
      <div className="panel">
        <h3>Your Power BI-ready datasets are available.</h3>
        <div className="download-list">
          <button
            className="primary-btn"
            onClick={() =>
              window.open(`${API_BASE}/api/download/water_demand_cleaned.csv`)
            }
          >
            water_demand_cleaned.csv
          </button>
          <button
            className="secondary-btn"
            onClick={() =>
              window.open(
                `${API_BASE}/api/download/water_demand_predictions.csv`,
              )
            }
          >
            water_demand_predictions.csv
          </button>
          <button
            className="secondary-btn"
            onClick={() =>
              window.open(`${API_BASE}/api/download/model_comparison.csv`)
            }
          >
            model_comparison.csv
          </button>
          <button
            className="secondary-btn"
            onClick={() =>
              window.open(`${API_BASE}/api/download/monthly_demand_summary.csv`)
            }
          >
            monthly_demand_summary.csv
          </button>
        </div>
      </div>
    </div>
  );

  const renderAbout = () => (
    <div className="page about-page">
      <div className="panel">
        <h3>About Project</h3>
        <p>
          <strong>Problem:</strong> Water demand changes over time and accurate
          planning is difficult.
        </p>
        <p>
          <strong>Solution:</strong> Use historical data and machine learning to
          forecast future water demand.
        </p>
        <p>
          <strong>Innovation:</strong> Combine data mining, machine-learning
          forecasting and interactive Power BI analytics in one practical
          decision-support system.
        </p>
        <p>
          <strong>Expected Outcome:</strong> Better understanding of demand
          patterns, future demand prediction and improved resource planning.
        </p>
        <p>
          <strong>Future Scope:</strong> IoT smart meters, real-time weather
          data, cloud deployment and real-time alerts.
        </p>
      </div>
    </div>
  );

  const renderActiveTab = () => {
    switch (activeTab) {
      case "Dataset":
        return renderDataset();
      case "Data Analysis":
        return renderAnalysis();
      case "Forecasting":
        return renderForecast();
      case "Model Comparison":
        return renderComparison();
      case "Power BI Export":
        return renderPowerbi();
      case "About Project":
        return renderAbout();
      case "Overview":
      default:
        return renderOverview();
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-icon">💧</div>
          <div>
            <strong>Water Demand</strong>
            <small>AI Dashboard</small>
          </div>
        </div>
        <nav className="nav">
          {navItems.map((item) => (
            <button
              key={item}
              className={activeTab === item ? "nav-link active" : "nav-link"}
              onClick={() => setActiveTab(item)}
            >
              {item}
            </button>
          ))}
        </nav>
      </header>

      {loading && <div className="loading-bar">Loading data...</div>}
      {errorMessage && <div className="error-message">{errorMessage}</div>}

      <main className="content-wrap">{renderActiveTab()}</main>
    </div>
  );
}

export default App;
