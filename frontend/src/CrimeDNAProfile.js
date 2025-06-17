// Simple React component to fetch and display a crime DNA profile

import React, { useState } from "react";

export default function CrimeDNAProfile() {
  const [crimeId, setCrimeId] = useState(0);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const fetchProfile = async () => {
    setLoading(true);
    setErr("");
    setProfile(null);
    try {
      const res = await fetch(`/crimes/dna_profile?crime_id=${crimeId}`);
      const data = await res.json();
      if (data.error) setErr(data.error);
      else setProfile(data);
    } catch (e) {
      setErr("Failed to fetch DNA profile.");
    }
    setLoading(false);
  };

  return (
    <div style={{ background: "#18181b", color: "#fff", borderRadius: 12, padding: 24, maxWidth: 700, margin: "32px auto", boxShadow: "0 2px 8px #222" }}>
      <h2 style={{ color: "#4ECDC4" }}>Crime DNA Profile Explorer</h2>
      <div style={{ marginBottom: 16 }}>
        <label>Crime Index:&nbsp;</label>
        <input
          type="number"
          value={crimeId}
          min={0}
          onChange={e => setCrimeId(Number(e.target.value))}
          style={{ width: 80, padding: 6, borderRadius: 6, border: "1px solid #333" }}
        />
        <button onClick={fetchProfile} style={{ marginLeft: 12, padding: "6px 18px", borderRadius: 6, background: "#4ECDC4", color: "#18181b", fontWeight: 600, border: "none" }}>
          Fetch DNA
        </button>
      </div>
      {loading && <div>Loading...</div>}
      {err && <div style={{ color: "#f87171" }}>{err}</div>}
      {profile && (
        <div style={{ marginTop: 18 }}>
          <div style={{ marginBottom: 10 }}>
            <b>Case #{profile.report_number}</b> — <span style={{ color: "#b45309" }}>{profile.crime_description}</span>
            <br />
            <span style={{ color: "#647dee" }}>{profile.city}</span> | Victim: {profile.victim_age}, {profile.victim_gender}
            <br />
            Weapon: {profile.weapon_used} | Domain: {profile.crime_domain} | Police: {profile.police_deployed} | Closed: {profile.case_closed}
          </div>
          <div>
            <b>DNA Vector:</b>
            <div
              className="dna-vector-scroll"
              style={{
                overflowX: "auto",
                marginTop: 8,
                background: "#fff",
                borderRadius: 10,
                padding: "10px 0",
                boxShadow: "0 1px 4px #e3e3e3",
                maxWidth: "100%",
              }}
            >
              <table
                className="dna-vector-table"
                style={{
                  width: "max-content",
                  minWidth: 480,
                  borderCollapse: "separate",
                  borderSpacing: "0 2px",
                  fontFamily: "'JetBrains Mono', 'Fira Mono', 'Consolas', 'Menlo', monospace",
                  fontSize: "1.05rem",
                  margin: "0 auto"
                }}
              >
                <thead>
                  <tr>
                    {profile.dna_features.map((f, i) => (
                      <th key={i}>{f}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    {profile.dna_vector.map((v, i) => (
                      <td
                        className="dna-vector-value"
                        key={i}
                        style={{
                          padding: "8px 16px",
                          borderRadius: 6,
                          background: "#f8fafc",
                          fontWeight: 700,
                          textAlign: "center",
                          minWidth: 80,
                          fontSize: "1.08rem",
                          lineHeight: "1.5"
                        }}
                      >
                        {v.toFixed(3)}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
