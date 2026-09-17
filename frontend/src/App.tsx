import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { api } from "./lib/api";
import { useAsync } from "./lib/useAsync";
import AgentRunsPage from "./pages/AgentRunsPage";
import DictionaryPage from "./pages/DictionaryPage";
import FomPage from "./pages/FomPage";
import MaterialDetailPage from "./pages/MaterialDetailPage";
import MaterialsPage from "./pages/MaterialsPage";
import MediationPage from "./pages/MediationPage";

const TABS = [
  { to: "/materials", label: "Materials" },
  { to: "/fom", label: "Figures of merit" },
  { to: "/mediation", label: "Mediated effects" },
  { to: "/runs", label: "Agent runs" },
  { to: "/dictionary", label: "Dictionary" },
];

export default function App() {
  const health = useAsync(() => api.health(), []);

  return (
    <div className="app">
      <header className="masthead">
        <h1>CNMS Living FOM</h1>
        <span className="subtitle">structure &rarr; property &rarr; function</span>
        <span style={{ marginLeft: "auto" }}>
          {health.data ? (
            <span className="badge ok">API v{health.data.version}</span>
          ) : health.error ? (
            <span className="badge bad">API unreachable</span>
          ) : (
            <span className="badge muted">connecting…</span>
          )}
        </span>
      </header>

      <nav className="tabs">
        {TABS.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            {tab.label}
          </NavLink>
        ))}
      </nav>

      <Routes>
        <Route path="/" element={<Navigate to="/materials" replace />} />
        <Route path="/materials" element={<MaterialsPage />} />
        <Route path="/materials/:id" element={<MaterialDetailPage />} />
        <Route path="/fom" element={<FomPage />} />
        <Route path="/mediation" element={<MediationPage />} />
        <Route path="/runs" element={<AgentRunsPage />} />
        <Route path="/dictionary" element={<DictionaryPage />} />
        <Route path="*" element={<div className="panel">Not found.</div>} />
      </Routes>
    </div>
  );
}
