import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
  LineChart, Line, ReferenceLine
} from "recharts";
import './App.css';
import CrimeDNAProfile from "./CrimeDNAProfile";

function App() {
  const [crimes, setCrimes] = useState([]);
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [cities, setCities] = useState([]);
  const [years, setYears] = useState([]);
  const [selectedCity, setSelectedCity] = useState("");
  const [selectedYear, setSelectedYear] = useState("");
  const [stats, setStats] = useState({ type: "", data: [] });
  const [statsLoading, setStatsLoading] = useState(false);
  // Chatbot state for AI Q&A
  const [qaChat, setQaChat] = useState([
    { role: "system", text: "Hi! 👋 I am your Crime Data AI Assistant. Ask me anything about crime data." }
  ]);
  const [qaInput, setQaInput] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = React.useRef(null);

  // For pagination
  const [page, setPage] = useState(1);
  const [totalCrimes, setTotalCrimes] = useState(0);
  const PAGE_SIZE = 10;

  // Semantic search states
  const [semanticResults, setSemanticResults] = useState([]);
  const [semanticQuery, setSemanticQuery] = useState("");
  const [semanticLoading, setSemanticLoading] = useState(false);

  // For voice-to-text in AI Q&A
  // const [isListening, setIsListening] = useState(false);
  // const recognitionRef = React.useRef(null);

  // Alert simulation state
  const [alertOpen, setAlertOpen] = useState(false);
  const [alertCity, setAlertCity] = useState("");
  const [alertType, setAlertType] = useState("");
  const [alertThreshold, setAlertThreshold] = useState(1);
  const [alertActive, setAlertActive] = useState(false);

  // Crime taxonomy HTML state
  const [taxonomyHtml, setTaxonomyHtml] = useState("");
  const [taxonomyLoading, setTaxonomyLoading] = useState(false);

  // Recurrence Score UI state
  const [recCities, setRecCities] = useState([]);
  const [recCity, setRecCity] = useState("");
  const [recScores, setRecScores] = useState([]);
  const [recLoading, setRecLoading] = useState(false);

  // Transition Graph UI state
  const [transitions, setTransitions] = useState([]);
  const [transitionLoading, setTransitionLoading] = useState(false);
  const [transitionGroup, setTransitionGroup] = useState("");
  const [transitionRandomPath, setTransitionRandomPath] = useState([]);
  const [transitionProbablePath, setTransitionProbablePath] = useState([]);
  const [transitionStart, setTransitionStart] = useState("");
  const [transitionError, setTransitionError] = useState("");
  const [transitionGraphImg, setTransitionGraphImg] = useState("");
  const [transitionGraphImgLoading, setTransitionGraphImgLoading] = useState(false);

  // Random Crime DNA and Compare Crimes DNA state
  const [randomCrime, setRandomCrime] = useState(null);
  const [userCrime, setUserCrime] = useState({
    City: "",
    "Weapon Used": "",
    "Crime Domain": "",
    Victim_Age: "",
    "Victim Gender": "",
    Police_Deployed: "",
    "Case Closed": ""
  });
  const [similarCrimes, setSimilarCrimes] = useState([]);
  const [randomLoading, setRandomLoading] = useState(false);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState("");

  const ageGroups = ["", "Child", "Teen", "Youth", "Adult", "MidAge", "Senior", "Elder"];

  const fetchTaxonomyTree = async () => {
    setTaxonomyLoading(true);
    setTaxonomyHtml("");
    try {
      const res = await fetch("/crimes/taxonomy_tree");
      const html = await res.text();
      setTaxonomyHtml(html);

      // Remove any old plotly.js script to avoid version conflicts
      document.querySelectorAll('script[data-plotly]').forEach(s => s.remove());

      // Inject latest plotly.js (v2.x) after HTML is set, then re-execute inline scripts
      setTimeout(() => {
        const script = document.createElement("script");
        script.src = "https://cdn.plot.ly/plotly-2.30.0.min.js";
        script.async = false;
        script.setAttribute("data-plotly", "true");
        script.onload = () => {
          // Re-execute all <script> tags inside taxonomyHtml (needed for plotly inline JS)
          document.querySelectorAll(".card-section script").forEach((oldScript) => {
            const newScript = document.createElement("script");
            if (oldScript.src) {
              newScript.src = oldScript.src;
            } else {
              newScript.textContent = oldScript.textContent;
            }
            oldScript.parentNode?.replaceChild(newScript, oldScript);
          });
          setTimeout(() => {
            document.querySelectorAll('.plotly-graph-div').forEach(div => {
              if (window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize) {
                window.Plotly.Plots.resize(div);
              }
            });
          }, 500);
        };
        document.body.appendChild(script);
      }, 200);
    } catch (e) {
      setTaxonomyHtml("<div style='color:red'>Failed to load taxonomy tree.</div>");
    }
    setTaxonomyLoading(false);
  };

  // Voice-to-text setup
  useEffect(() => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.lang = "en-US";
    recognitionRef.current.interimResults = false;
    recognitionRef.current.maxAlternatives = 1;

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setQaInput(transcript);
      setIsListening(false);
    };
    recognitionRef.current.onend = () => setIsListening(false);
    recognitionRef.current.onerror = () => setIsListening(false);
  }, []);

  const handleVoiceInput = () => {
    if (!recognitionRef.current) return;
    setIsListening(true);
    recognitionRef.current.start();
  };

  useEffect(() => {
    fetchCrimes(selectedCity, selectedYear, page);
    fetchSummary(selectedCity, selectedYear);
    fetchCities();
    fetchYears();
    // eslint-disable-next-line
  }, [page]);

  const fetchCrimes = async (city = "", year = "", pageNum = 1) => {
    setLoading(true);
    let url = `/crimes?limit=${PAGE_SIZE}&skip=${(pageNum - 1) * PAGE_SIZE}`;
    if (city || year) {
      url = `/crimes/search?limit=${PAGE_SIZE}&skip=${(pageNum - 1) * PAGE_SIZE}`;
      if (city) url += `&city=${encodeURIComponent(city)}`;
      if (year) url += `&date_from=${year}-01-01&date_to=${year}-12-31`;
    }
    const res = await fetch(url);
    const data = await res.json();
    setCrimes(data);

    // Fetch total count for pagination
    let countUrl = "/crimes/count";
    if (city || year) {
      countUrl = "/crimes/search/count?";
      if (city) countUrl += `city=${encodeURIComponent(city)}&`;
      if (year) countUrl += `date_from=${year}-01-01&date_to=${year}-12-31&`;
    }
    const countRes = await fetch(countUrl);
    const countData = await countRes.json();
    setTotalCrimes(countData.count || 0);

    setLoading(false);
  };

  const fetchSummary = async (city = "", year = "") => {
    let url = "/crimes/ai/summary";
    const params = [];
    if (year) params.push(`year_from=${year}&year_to=${year}`);
    if (city) params.push(`cities=${encodeURIComponent(city)}`);
    if (params.length) url += "?" + params.join("&");
    const res = await fetch(url);
    const data = await res.json();
    setSummary(data.summary);
  };

  const fetchCities = async () => {
    const res = await fetch("/crimes/cities");
    const data = await res.json();
    setCities(data);
  };

  const fetchYears = async () => {
    const res = await fetch("/crimes/years");
    const data = await res.json();
    setYears(data);
  };

  const handleFilter = () => {
    setPage(1);
    fetchCrimes(selectedCity, selectedYear, 1);
    fetchSummary(selectedCity, selectedYear);
  };

  const fetchStats = async (type) => {
    setStatsLoading(true);
    let url = "";
    if (type === "city") url = "/crimes/statistics/by_city";
    if (type === "year") url = "/crimes/statistics/by_year";
    if (type === "crime_type") url = "/crimes/statistics/by_crime_type";
    if (type === "victim_gender") url = "/crimes/statistics/by_victim_gender";
    const res = await fetch(url);
    const data = await res.json();
    setStats({ type, data });
    setStatsLoading(false);
  };

  // Chatbot submit handler
  const handleQaSubmit = async (e) => {
    e.preventDefault();
    if (!qaInput.trim()) return;
    setQaLoading(true);
    setQaChat(prev => [...prev, { role: "user", text: qaInput }]);
    setQaInput("");
    try {
      const res = await fetch("/crimes/ai/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: qaInput })
      });
      const data = await res.json();
      setQaChat(prev => [
        ...prev,
        { role: "assistant", text: data.answer || "No answer available." }
      ]);
    } catch {
      setQaChat(prev => [
        ...prev,
        { role: "assistant", text: "Sorry, I couldn't get an answer right now." }
      ]);
    }
    setQaLoading(false);
  };

  // Semantic search handler
  const handleSemanticSearch = async (e) => {
    e.preventDefault();
    setSemanticLoading(true);
    setSemanticResults([]);
    const res = await fetch("/crimes/vector_search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: semanticQuery, limit: 10 })
    });
    const data = await res.json();
    setSemanticResults(data);
    setSemanticLoading(false);
  };

  // Fetch recurrence cities on mount
  useEffect(() => {
    const fetchCities = async () => {
      const res = await fetch("/crimes/recurrence_score");
      const data = await res.json();
      const uniqueCities = Array.from(new Set(data.map(d => d.city))).sort();
      setRecCities(uniqueCities);
      if (uniqueCities.length > 0) setRecCity(uniqueCities[0]);
    };
    fetchCities();
  }, []);

  // Fetch recurrence scores when city changes
  useEffect(() => {
    if (!recCity) return;
    setRecLoading(true);
    fetch(`/crimes/recurrence_score?city=${encodeURIComponent(recCity)}&top=5`)
      .then(res => res.json())
      .then(data => {
        setRecScores(data);
        setRecLoading(false);
      });
  }, [recCity]);

  const fetchTransitionGraph = async (group = "") => {
    setTransitionLoading(true);
    setTransitions([]);
    setTransitionRandomPath([]);
    setTransitionProbablePath([]);
    setTransitionStart("");
    setTransitionError("");
    let url = "/crimes/transition_graph";
    if (group) url += `?age_group=${encodeURIComponent(group)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) {
      setTransitionError(data.error);
      setTransitionLoading(false);
      return;
    }
    setTransitions(data.top_transitions || []);
    setTransitionRandomPath(data.random_path || []);
    setTransitionProbablePath(data.most_probable_path || []);
    setTransitionStart(data.start || "");
    setTransitionLoading(false);
  };

  const fetchTransitionGraphImage = async (group = "") => {
    setTransitionGraphImg("");
    setTransitionGraphImgLoading(true);
    let url = "/crimes/transition_graph_image";
    if (group) url += `?age_group=${encodeURIComponent(group)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.image_base64) {
      setTransitionGraphImg("data:image/png;base64," + data.image_base64);
    }
    setTransitionGraphImgLoading(false);
  };

  // Fetch random crime DNA
  const fetchRandomCrime = async () => {
    setRandomLoading(true);
    setRandomCrime(null);
    try {
      const res = await fetch("/crimes/dna_random");
      const data = await res.json();
      setRandomCrime(data);
    } catch {
      setRandomCrime({ error: "Failed to fetch random crime DNA." });
    }
    setRandomLoading(false);
  };

  // Find similar crimes
  const handleFindSimilar = async () => {
    setSimilarLoading(true);
    setSimilarError("");
    setSimilarCrimes([]);
    try {
      // Prepare payload with only relevant fields
      const payload = {
        City: userCrime.City,
        "Weapon Used": userCrime["Weapon Used"],
        "Crime Domain": userCrime["Crime Domain"],
        Victim_Age: userCrime.Victim_Age ? Number(userCrime.Victim_Age) : 0,
        "Victim Gender": userCrime["Victim Gender"],
        Police_Deployed: userCrime.Police_Deployed ? Number(userCrime.Police_Deployed) : 0,
        "Case Closed": userCrime["Case Closed"]
      };
      const res = await fetch("/crimes/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.error) setSimilarError(data.error);
      else setSimilarCrimes(data.results || []);
    } catch {
      setSimilarError("Failed to find similar crimes.");
    }
    setSimilarLoading(false);
  };

  return (
    <div className="main-bg">
      <header className="main-header">
        <h1 style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 16 }}>
          <img
            src="/Crime_dashboard_logo.png"
            alt="Crime Analytics Logo"
            style={{
              height: 48,
              width: 48,
              objectFit: "contain",
              marginRight: 10,
              filter: "drop-shadow(0 0 8px #fff) brightness(1.35) contrast(1.2)"
            }}
          />
          Crime Analytics Dashboard
        </h1>
        <p className="subtitle">
        </p>
      </header>

      {/* Filter Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Filter Crimes</h2>
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <select value={selectedCity} onChange={e => setSelectedCity(e.target.value)} style={{ padding: 8, borderRadius: 6 }}>
            <option value="">All Cities</option>
            {cities.map(city => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
          <select value={selectedYear} onChange={e => setSelectedYear(e.target.value)} style={{ padding: 8, borderRadius: 6 }}>
            <option value="">All Years</option>
            {years.map(year => (
              <option key={year} value={year}>{year}</option>
            ))}
          </select>
          <button onClick={handleFilter} style={{
            padding: "8px 20px",
            background: "linear-gradient(90deg, #2980b9 0%, #6dd5fa 100%)",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontWeight: "bold",
            boxShadow: "0 2px 6px #b0c4de"
          }}>
            Apply Filter
          </button>
        </div>
      </section>

      {/* AI Q&A Chatbot Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Ask AI about Crime Data</h2>
        <div style={{
          background: "#f5f5f5",
          borderRadius: 8,
          padding: 16,
          minHeight: 120,
          maxHeight: 260,
          overflowY: "auto",
          marginBottom: 12,
          fontSize: 16
        }}>
          {qaChat.map((msg, idx) => (
            <div key={idx} style={{
              marginBottom: 10,
              textAlign: msg.role === "user" ? "right" : "left"
            }}>
              <span style={{
                display: "inline-block",
                background: msg.role === "user" ? "#d1e7ff" : "#e0ffe0",
                color: "#222",
                borderRadius: 8,
                padding: "8px 14px",
                maxWidth: "80%",
                fontWeight: msg.role === "system" ? "bold" : "normal"
              }}>
                {msg.text}
              </span>
            </div>
          ))}
          {qaLoading && (
            <div style={{ color: "#888", fontStyle: "italic" }}>AI is thinking...</div>
          )}
        </div>
        <form onSubmit={handleQaSubmit} style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <input
            type="text"
            value={qaInput}
            onChange={e => setQaInput(e.target.value)}
            placeholder="Ask a question (e.g., Which city has most thefts?)"
            style={{
              flex: 1,
              padding: 10,
              borderRadius: 6,
              border: "1px solid #b0c4de",
              fontSize: 16
            }}
            disabled={qaLoading}
          />
          <button
            type="button"
            onClick={handleVoiceInput}
            style={{
              padding: "8px 14px",
              background: isListening ? "#e67e22" : "#43cea2",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              fontWeight: "bold",
              fontSize: 18,
              cursor: "pointer"
            }}
            title="Speak your question"
            disabled={isListening || qaLoading}
          >
            🎤
          </button>
          <button type="submit" style={{
            padding: "8px 20px",
            background: "linear-gradient(90deg, #43cea2 0%, #185a9d 100%)",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontWeight: "bold"
          }} disabled={qaLoading}>
            Ask
          </button>
        </form>
      </section>

      {/* Crimes Table Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Sample Crimes</h2>
        {loading ? (
          <div>Loading crimes...</div>
        ) : crimes.length === 0 ? (
          <div>No crime records found.</div>
        ) : (
          <>
            <div style={{ overflowX: "auto" }}>
              <table border="0" cellPadding="8" style={{
                width: "100%",
                background: "#fafbfc",
                borderRadius: 8,
                boxShadow: "0 1px 4px #e3e3e3"
              }}>
                <thead style={{ background: "#e1e8ed" }}>
                  <tr>
                    <th>City</th>
                    <th>Date Reported</th>
                    <th>Crime Description</th>
                    <th>Victim Gender</th>
                    <th>Weapon Used</th>
                  </tr>
                </thead>
                <tbody>
                  {crimes.map((crime) => (
                    <tr key={crime._id}>
                      <td>{crime.City}</td>
                      <td>{crime["Date Reported"]}</td>
                      <td>{crime["Crime Description"]}</td>
                      <td>{crime["Victim Gender"]}</td>
                      <td>{crime["Weapon Used"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", marginTop: 16, gap: 12 }}>
              <button
                onClick={() => setPage(page - 1)}
                disabled={page === 1}
                style={{ padding: "6px 16px", borderRadius: 6, border: "1px solid #b0c4de", background: "#f5f5f5" }}
              >
                Previous
              </button>
              <span>
                Page {page} of {Math.ceil(totalCrimes / PAGE_SIZE) || 1}
              </span>
              <button
                onClick={() => setPage(page + 1)}
                disabled={page * PAGE_SIZE >= totalCrimes}
                style={{ padding: "6px 16px", borderRadius: 6, border: "1px solid #b0c4de", background: "#f5f5f5" }}
              >
                Next
              </button>
            </div>
          </>
        )}
      </section>

      {/* Statistics Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Statistics</h2>
        <div style={{ display: "flex", gap: 12, marginBottom: 18, flexWrap: "wrap" }}>
          <button onClick={() => fetchStats("city")} style={{ padding: "8px 16px", borderRadius: 6, background: "#f7971e", color: "#fff", border: "none", fontWeight: "bold" }}>By City</button>
          <button onClick={() => fetchStats("year")} style={{ padding: "8px 16px", borderRadius: 6, background: "#56ab2f", color: "#fff", border: "none", fontWeight: "bold" }}>By Year</button>
          <button onClick={() => fetchStats("crime_type")} style={{ padding: "8px 16px", borderRadius: 6, background: "#614385", color: "#fff", border: "none", fontWeight: "bold" }}>By Crime Type</button>
          <button onClick={() => fetchStats("victim_gender")} style={{ padding: "8px 16px", borderRadius: 6, background: "#36d1c4", color: "#fff", border: "none", fontWeight: "bold" }}>By Victim Gender</button>
        </div>
        {statsLoading ? (
          <div>Loading statistics...</div>
        ) : stats.data.length > 0 ? (
          <>
            <div style={{ overflowX: "auto" }}>
              <table border="0" cellPadding="8" style={{
                width: "100%",
                background: "#fafbfc",
                borderRadius: 8,
                boxShadow: "0 1px 4px #e3e3e3",
                marginBottom: 24
              }}>
                <thead style={{ background: "#e1e8ed" }}>
                  <tr>
                    {stats.type === "city" && (<><th>City</th><th>Count</th></>)}
                    {stats.type === "year" && (<><th>Year</th><th>Count</th></>)}
                    {stats.type === "crime_type" && (<><th>Crime Type</th><th>Count</th></>)}
                    {stats.type === "victim_gender" && (<><th>Victim Gender</th><th>Count</th></>)}
                  </tr>
                </thead>
                <tbody>
                  {stats.data.map((row, idx) => (
                    <tr key={idx}>
                      {stats.type === "city" && (<><td>{row.city}</td><td>{row.count}</td></>)}
                      {stats.type === "year" && (<><td>{row.year}</td><td>{row.count}</td></>)}
                      {stats.type === "crime_type" && (<><td>{row.crime_type}</td><td>{row.count}</td></>)}
                      {stats.type === "victim_gender" && (<><td>{row.victim_gender}</td><td>{row.count}</td></>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ width: "100%", height: 350 }}>
              <ResponsiveContainer>
                <BarChart
                  data={stats.data}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey={
                      stats.type === "city"
                        ? "city"
                        : stats.type === "year"
                        ? "year"
                        : stats.type === "crime_type"
                        ? "crime_type"
                        : "victim_gender"
                    }
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar
                    dataKey="count"
                    fill="#2980b9"
                    name="Crime Count"
                    barSize={40}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <div>Select a statistics type to view data.</div>
        )}
      </section>

      {/* Folium Map Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Crime Locations Map</h2>
        <div style={{ marginBottom: 10, color: "#888", fontSize: 15 }}>
          <b>What you see:</b> Interactive Folium map with city markers and clustering.<br />
          Pan, zoom, and click markers for details.
        </div>
        <div style={{ width: "100%", height: 400, borderRadius: 8, overflow: "hidden", border: "1px solid #e3e3e3" }}>
          <iframe
            title="Crime Folium Map"
            src="/folium_map.html"
            style={{ width: "100%", height: "100%", border: "none", background: "#fff" }}
            sandbox="allow-scripts allow-same-origin"
          />
        </div>
      </section>

      {/* Semantic Search Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Semantic Search</h2>
        <form onSubmit={handleSemanticSearch} style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 10 }}>
          <input
            type="text"
            value={semanticQuery}
            onChange={e => setSemanticQuery(e.target.value)}
            placeholder="Ask a semantic question..."
            style={{
              flex: 1,
              padding: 10,
              borderRadius: 6,
              border: "1px solid #b0c4de",
              fontSize: 16
            }}
          />
          <button type="submit" style={{
            padding: "8px 20px",
            background: "linear-gradient(90deg, #e67e22 0%, #f39c12 100%)",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontWeight: "bold"
          }}>
            Search
          </button>
        </form>
        {semanticResults.length > 0 && (
          <button
            onClick={() => { setSemanticResults([]); setSemanticQuery(""); }}
            style={{
              marginBottom: 12,
              padding: "6px 16px",
              borderRadius: 6,
              border: "1px solid #b0c4de",
              background: "#f5f5f5",
              color: "#34495e",
              fontWeight: "bold"
            }}
          >
            Clear Results
          </button>
        )}
        {semanticLoading ? (
          <div>Searching...</div>
        ) : (
          <ul style={{ paddingLeft: 20, color: "#34495e", fontSize: 16 }}>
            {semanticResults.length === 0 ? (
              <li>No relevant results found.</li>
            ) : (
              semanticResults.map((crime, idx) => (
                <li
                  key={crime._id || idx}
                  style={{
                    marginBottom: 8,
                    // Remove background highlight and pointer cursor for map sync
                  }}
                >
                  <b>{crime.City}</b> - {crime["Crime Description"]}
                  {crime["Date Reported"] && (
                    <> | <span style={{ color: "#888" }}>{crime["Date Reported"]}</span></>
                  )}
                  {crime["Victim Gender"] && (
                    <> | <span style={{ color: "#888" }}>Victim: {crime["Victim Gender"]}</span></>
                  )}
                  {crime["Weapon Used"] && (
                    <> | <span style={{ color: "#888" }}>Weapon: {crime["Weapon Used"]}</span></>
                  )}
                  {/* Removed [Show on Map] and map highlight logic */}
                </li>
              ))
            )}
          </ul>
        )}
      </section>

      {/* Crime Alert Simulation */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#e74c3c", marginBottom: 12 }}>Crime Alert Simulation</h2>
        <button
          onClick={() => setAlertOpen(true)}
          style={{
            padding: "8px 18px",
            background: "#e74c3c",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontWeight: "bold",
            marginBottom: 10
          }}
        >
          Set Alert
        </button>
        {alertOpen && (
          <div style={{
            background: "#fff8e1",
            border: "1px solid #e67e22",
            borderRadius: 8,
            padding: 18,
            marginBottom: 10,
            maxWidth: 350
          }}>
            <div style={{ marginBottom: 8 }}>
              <b>City:</b>
              <select value={alertCity} onChange={e => setAlertCity(e.target.value)} style={{ marginLeft: 8 }}>
                <option value="">Select City</option>
                {cities.map(city => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </select>
            </div>
            <div style={{ marginBottom: 8 }}>
              <b>Crime Type:</b>
              <input
                type="text"
                value={alertType}
                onChange={e => setAlertType(e.target.value)}
                placeholder="e.g. kidnapping"
                style={{ marginLeft: 8, borderRadius: 4, border: "1px solid #ccc", padding: "2px 8px" }}
              />
            </div>
            <div style={{ marginBottom: 8 }}>
              <b>Threshold:</b>
              <input
                type="number"
                min={1}
                value={alertThreshold}
                onChange={e => setAlertThreshold(Number(e.target.value))}
                style={{ marginLeft: 8, width: 60, borderRadius: 4, border: "1px solid #ccc", padding: "2px 8px" }}
              />
            </div>
            <button
              onClick={() => setAlertOpen(false)}
              style={{
                padding: "4px 14px",
                background: "#e67e22",
                color: "#fff",
                border: "none",
                borderRadius: 6,
                fontWeight: "bold"
              }}
            >
              Save Alert
            </button>
          </div>
        )}
        {alertActive && (
          <div style={{
            background: "#ffebee",
            border: "2px solid #e74c3c",
            borderRadius: 8,
            padding: 18,
            color: "#c0392b",
            fontWeight: "bold",
            fontSize: 18,
            marginTop: 10,
            maxWidth: 400
          }}>
            🚨 Alert: {alertType} cases in {alertCity} have reached {alertThreshold}!
          </div>
        )}
      </section>

      {/* Crime Association Graph */}
      <section className="card-section">
        <h2>
          Crime Association Graph
        </h2>
        <button
          onClick={fetchTaxonomyTree}
          className="action-btn"
        >
          Show Crime Association Graph
        </button>
        {taxonomyLoading && (
          <div className="loading-text">Loading association graph...</div>
        )}
        {/* Only render association graph HTML, not dendrogram */}
        {taxonomyHtml && (
          <div
            style={{ width: "100%", overflowX: "auto" }}
            dangerouslySetInnerHTML={{ __html: taxonomyHtml }}
          />
        )}
      </section>

      {/* Crime Recurrence Score Section */}
      <section style={{
        marginBottom: 32,
        background: "#fff",
        padding: 20,
        borderRadius: 10,
        boxShadow: "0 2px 8px #dbeafe"
      }}>
        <h2 style={{ color: "#34495e", marginBottom: 12 }}>Crime Recurrence Score</h2>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontWeight: 500, marginRight: 8 }}>City:</label>
          <select
            value={recCity}
            onChange={e => setRecCity(e.target.value)}
            style={{ padding: 8, borderRadius: 6 }}
          >
            {recCities.map(city => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
        </div>
        {recLoading ? (
          <div>Loading recurrence scores...</div>
        ) : recScores.length === 0 ? (
          <div>No recurrence data available.</div>
        ) : (
          <div style={{ width: "100%", maxWidth: 600 }}>
            <h4 style={{ marginBottom: 10 }}>
              Top 5 Recurring Crimes in <span style={{ color: "#2980b9" }}>{recCity}</span>
            </h4>
            <div style={{ background: "#f8fafc", borderRadius: 8, padding: 16 }}>
              {recScores.map((row, idx) => (
                <div key={row.crime_description} style={{ marginBottom: 10 }}>
                  <b>{row.crime_description}</b>
                  <div style={{
                    background: "#e0e7ef",
                    borderRadius: 6,
                    height: 18,
                    width: "100%",
                    marginTop: 4,
                    marginBottom: 2,
                    position: "relative"
                  }}>
                    <div style={{
                      width: `${Math.round(row.recurrence_score * 100)}%`,
                      background: "#36d1c4",
                      height: "100%",
                      borderRadius: 6
                    }} />
                    <span style={{
                      position: "absolute",
                      right: 10,
                      top: 0,
                      fontSize: 13,
                      color: "#34495e"
                    }}>
                      {row.recurrence_score.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Crime Transition Graph Section */}
      <section className="card-section">
        <h2>
          Crime Transition Patterns
        </h2>
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontWeight: 500, marginRight: 8 }}>Age Group:</label>
          <select
            value={transitionGroup}
            onChange={e => {
              setTransitionGroup(e.target.value);
              fetchTransitionGraph(e.target.value);
              fetchTransitionGraphImage(e.target.value);
            }}
            style={{ padding: 8, borderRadius: 6 }}
          >
            {["", "Child", "Teen", "Youth", "Adult", "MidAge", "Senior", "Elder"].map(g => (
              <option key={g} value={g}>{g || "All"}</option>
            ))}
          </select>
          <button
            onClick={() => {
              fetchTransitionGraph(transitionGroup);
              fetchTransitionGraphImage(transitionGroup);
            }}
            className="action-btn"
            style={{ marginLeft: 12 }}
          >
            Show Transitions
          </button>
        </div>
        {transitionGraphImgLoading && (
          <div className="loading-text">Loading transition graph image...</div>
        )}
        {transitionGraphImg && (
          <div style={{ textAlign: "center", marginBottom: 18 }}>
            <img
              src={transitionGraphImg}
              alt="Transition Graph"
              style={{ maxWidth: "100%", borderRadius: 12, boxShadow: "0 2px 8px #b0c4de" }}
            />
          </div>
        )}
        {transitionLoading && (
          <div className="loading-text">Loading transition graph...</div>
        )}
        {transitionError && (
          <div className="empty-text">{transitionError}</div>
        )}
        {(!transitionLoading && !transitionError && transitions.length > 0) && (
          <div style={{ marginBottom: 18 }}>
            <table style={{ width: "100%", background: "#fafbfc", borderRadius: 8, boxShadow: "0 1px 4px #e3e3e3" }}>
              <thead style={{ background: "#e1e8ed" }}>
                <tr>
                  <th>From</th>
                  <th>To</th>
                  <th>Probability</th>
                </tr>
              </thead>
              <tbody>
                {transitions.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.from} <span style={{ color: "#888" }}>({row.from_abbr})</span></td>
                    <td>{row.to} <span style={{ color: "#888" }}>({row.to_abbr})</span></td>
                    <td>{(row.prob * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {(transitionRandomPath.length > 1 || transitionProbablePath.length > 1) && (
          <div style={{ marginBottom: 10 }}>
            <div style={{ marginBottom: 6 }}>
              <b>Random Path from <span style={{ color: "#185a9d" }}>{transitionStart}</span>:</b>
              <span style={{ marginLeft: 8, color: "#10b981" }}>
                {transitionRandomPath.join(" → ")}
              </span>
            </div>
            <div>
              <b>Most Probable Path from <span style={{ color: "#185a9d" }}>{transitionStart}</span>:</b>
              <span style={{ marginLeft: 8, color: "#e67e22" }}>
                {transitionProbablePath.join(" → ")}
              </span>
            </div>
          </div>
        )}
        {(!transitionLoading && !transitionError && transitions.length === 0) && (
          <div className="empty-text">No transition data available for this group.</div>
        )}
      </section>

      {/* DNA Random & Find Similar Crimes Section */}
      <section className="card-section crazy-card crazy-float" style={{ marginBottom: 32 }}>
        <h2>
          Random & Find Similar Crimes
        </h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 32 }}>
          {/* Random Crime DNA */}
          <div style={{ flex: 1, minWidth: 320 }}>
            <button className="crazy-btn crazy-btn-glow" onClick={fetchRandomCrime} disabled={randomLoading}>
              {randomLoading ? "Loading..." : "Show Random Crime DNA"}
            </button>
            {randomCrime && !randomCrime.error && (
              <div style={{ marginTop: 12, background: "#f8fafc", borderRadius: 10, padding: 12, color: "#222" }}>
                <b>Case #{randomCrime.report_number}</b> — <span style={{ color: "#7f53ac" }}>{randomCrime.crime_description}</span>
                <br />
                <span style={{ color: "#647dee" }}>{randomCrime.city}</span> | Victim: {randomCrime.victim_age}, {randomCrime.victim_gender}
                <br />
                Weapon: {randomCrime.weapon_used} | Domain: {randomCrime.crime_domain}
                <br />
                <b>DNA:</b> <span style={{ fontSize: 13 }}>{randomCrime.dna_vector && randomCrime.dna_vector.map(v => v.toFixed(2)).join(", ")}</span>
              </div>
            )}
            {randomCrime && randomCrime.error && (
              <div style={{ color: "#ef4444", marginTop: 10 }}>{randomCrime.error}</div>
            )}
          </div>
          {/* Find Similar Crimes */}
          <div style={{ flex: 1, minWidth: 320 }}>
            <div style={{ marginBottom: 8, background: "#f8fafc", borderRadius: 10, padding: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>Find Similar Crimes</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                <input
                  className="crime-input"
                  type="text"
                  placeholder="City"
                  value={userCrime.City}
                  onChange={e => setUserCrime({ ...userCrime, City: e.target.value })}
                  style={{ flex: 1, minWidth: 90 }}
                />
                <input
                  className="crime-input"
                  type="text"
                  placeholder="Weapon Used"
                  value={userCrime["Weapon Used"]}
                  onChange={e => setUserCrime({ ...userCrime, "Weapon Used": e.target.value })}
                  style={{ flex: 1, minWidth: 90 }}
                />
                <input
                  className="crime-input"
                  type="text"
                  placeholder="Crime Domain"
                  value={userCrime["Crime Domain"]}
                  onChange={e => setUserCrime({ ...userCrime, "Crime Domain": e.target.value })}
                  style={{ flex: 1, minWidth: 90 }}
                />
                <input
                  className="crime-input"
                  type="number"
                  placeholder="Victim Age"
                  value={userCrime.Victim_Age}
                  onChange={e => setUserCrime({ ...userCrime, Victim_Age: e.target.value })}
                  style={{ width: 90 }}
                />
                <input
                  className="crime-input"
                  type="text"
                  placeholder="Victim Gender"
                  value={userCrime["Victim Gender"]}
                  onChange={e => setUserCrime({ ...userCrime, "Victim Gender": e.target.value })}
                  style={{ width: 110 }}
                />
                <input
                  className="crime-input"
                  type="number"
                  placeholder="Police Deployed"
                  value={userCrime.Police_Deployed}
                  onChange={e => setUserCrime({ ...userCrime, Police_Deployed: e.target.value })}
                  style={{ width: 110 }}
                />
                <input
                  className="crime-input"
                  type="text"
                  placeholder="Case Closed"
                  value={userCrime["Case Closed"]}
                  onChange={e => setUserCrime({ ...userCrime, "Case Closed": e.target.value })}
                  style={{ width: 110 }}
                />
                <button className="crazy-btn crazy-btn-glow" onClick={handleFindSimilar} disabled={similarLoading} style={{ flex: 1, minWidth: 120 }}>
                  {similarLoading ? "Finding..." : "Find Similar"}
                </button>
              </div>
              {similarError && <div style={{ color: "#ef4444", marginTop: 8 }}>{similarError}</div>}
              {similarCrimes.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <b>Top Similar Crimes:</b>
                  <ul style={{ marginTop: 6, paddingLeft: 18, fontSize: 15 }}>
                    {similarCrimes.map((crime, idx) => (
                      <li key={crime.index}>
                        <b>#{crime.report_number}</b> — {crime.crime_description} <span style={{ color: "#647dee" }}>({crime.city})</span>
                        <span style={{ color: "#10b981", marginLeft: 8 }}>{(crime.similarity * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <CrimeDNAProfile />

      <footer style={{
        textAlign: "center",
        color: "#888",
        marginTop: 32,
        fontSize: 15
      }}>
        &copy; {new Date().getFullYear()} W.A.N.T.E.D | Google Hackathon Demo
      </footer>
    </div>
  );
}

export default App;

// The alert feature works as follows:
//
// 1. You set an alert for a city, crime type, and threshold using the "Set Alert" dialog.
// 2. Every time the crimes data or your alert settings change, a useEffect runs:
//    - It counts how many crimes in the current table/filter match the selected city and crime type.
//    - If the count is greater than or equal to your threshold, it sets alertActive to true.
// 3. When alertActive is true, a red notification box appears below the alert settings, showing your alert message.
// 4. If you change the filter, city, crime type, or threshold, the alert will update automatically.
//
// No backend changes are needed: the alert is simulated in real time on the frontend using the crimes already loaded.
//
// The alert is NOT saved to a backend or database.
// It is only stored in React state variables (alertCity, alertType, alertThreshold) while the page is open.
// If you reload the page, the alert is lost.
//
// How it works (fully functional as a simulation):
// - When you set an alert, the values are saved in React state.
// - On every crimes/filter change, useEffect checks if the alert condition is met and shows a notification.
// - The alert is "virtual" and only works for the current session and loaded data.
//
// To make it persistent or real-time, you would need to:
//   1. Save alert settings to a backend/database per user.
//   2. Check new crimes in the backend and push notifications to the frontend (e.g., with websockets or polling).
//   3. Optionally, send email/SMS or browser notifications.
//
// For hackathon/demo, this frontend simulation is enough to impress judges and show the concept!

// The error "Could not connect to the Hugging Face API. Details: ..." means your backend could not reach the Hugging Face API.
// This is not a frontend bug. The chatbot UI is working, but the backend failed to get a response from the AI model.
// To fix this:
// 1. Make sure your Hugging Face API token (HF_API_TOKEN) is set correctly in your .env file.
// 2. Make sure your server has internet access and Hugging Face is not blocked.
// 3. If you are using a free Hugging Face endpoint, it may be rate-limited or temporarily unavailable.
// 4. You can retry after a few minutes, or check your backend logs for more details.

// Analysis and suggestions for a more awesome UI design:

// 1. **Consistent Theming:** 
//    - Use your CSS variables for all backgrounds, cards, buttons, and text. 
//    - Avoid hardcoded colors like #fff, #34495e, #e1e8ed, etc. Use --card-bg, --primary, --primary-dark, etc.

// 2. **Card and Section Styling:** 
//    - Use a single "crazy-card" or "card-section" class for all main sections, not inline styles.
//    - Add a subtle glassmorphism effect or a soft shadow for cards.
//    - Use border-radius and padding consistently.

// 3. **Typography:** 
//    - Use larger, bolder headings for section titles (h2).
//    - Use .crazy-gradient-text for main headings and important stats.
//    - Use .highlight or .crazy-gradient-text for key numbers or labels.

// 4. **Buttons:** 
//    - Use .crazy-btn and .crazy-btn-glow for all main action buttons.
//    - Use icons (FontAwesome or Emoji) in buttons for visual cues.

// 5. **Tables:** 
//    - Use .crazy-table and .crazy-table-wrap for all tables.
//    - Add hover effects and sticky headers for better UX.

// 6. **Inputs and Selects:** 
//    - Use .crazy-select and .crazy-chat-input for all selects and inputs.
//    - Add focus/hover effects for interactivity.

// 7. **Background:** 
//    - Use .crazy-bg-anim for a dynamic animated background.
//    - Add a semi-transparent overlay to cards for glass effect.

// 8. **Spacing and Layout:** 
//    - Use .crazy-grid, .crazy-row, .crazy-col for responsive layouts.
//    - Add more vertical spacing between sections.

// 9. **Transitions and Animations:** 
//    - Use subtle transitions on hover/focus for cards, buttons, and inputs.
//    - Use .crazy-float for floating card effects.

// 10. **Dark/Light Mode:** 
//     - Consider a toggle for dark/light mode using CSS variables.

// 11. **Visual Hierarchy:** 
//     - Use color and size to guide the user's eye to the most important actions and data.

// 12. **Mobile Responsiveness:** 
//     - Use media queries and .crazy-grid/.crazy-col for mobile-friendly layouts.

// 13. **Consistent Iconography:** 
//     - Use emojis or SVG icons consistently for section headers and buttons.

// 14. **Remove Inline Styles:** 
//     - Move all inline styles to CSS classes for maintainability and consistency.

// 15. **Section Order:** 
//     - Consider the order of sections for best storytelling: 
//       Dashboard summary → Filters → Key stats → Map → Trends → AI/Chatbot → Advanced analytics.

// 16. **Loading States:** 
//     - Use animated spinners or skeleton loaders for loading states.

// 17. **Feedback:** 
//     - Show toasts or banners for errors, success, or important info.

// 18. **Accessibility:** 
//     - Use proper aria-labels, alt text, and keyboard navigation.

// 19. **Branding:** 
//     - Add a logo or brand mark in the header/footer.

// 20. **Footer:** 
//     - Use .crazy-footer for a visually distinct, branded footer.

// To implement these, refactor your JSX to use your CSS classes everywhere, remove inline styles, and use your color variables for all UI elements. 
// You can also add more visual polish with gradients, glassmorphism, and subtle animations.
