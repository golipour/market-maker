import { NavLink, Outlet } from "react-router-dom";
import { useState } from "react";
import { getToken, setToken } from "../api/client";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/strategies", label: "Strategies" },
  { to: "/allocations", label: "Allocations" },
  { to: "/workers", label: "Workers" },
  { to: "/commands", label: "Commands" },
  { to: "/observability", label: "Observability" },
];

export function Layout() {
  const [token, setTokenState] = useState(getToken());

  function saveToken(value: string) {
    setToken(value);
    setTokenState(value);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">Market Maker</div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => (isActive ? "active" : undefined)}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <label htmlFor="admin-token" style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Admin token
          </label>
          <input
            id="admin-token"
            className="token-input"
            type="password"
            placeholder="MM_ADMIN_TOKEN (optional locally)"
            value={token}
            onChange={(e) => saveToken(e.target.value)}
            style={{ marginTop: "0.35rem" }}
          />
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
