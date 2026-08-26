import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { getToken, setToken } from "../api/client";
import { ObservabilityPage } from "../pages/Observability";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/exchanges", label: "Exchanges" },
  { to: "/accounts", label: "Accounts" },
  { to: "/presets", label: "Presets" },
  { to: "/feeds", label: "Market data" },
  { to: "/strategies", label: "Strategies" },
  { to: "/workers", label: "Workers" },
  { to: "/settings", label: "Settings" },
  { to: "/commands", label: "Commands" },
  { to: "/observability", label: "Observability" },
];

export function Layout() {
  const { pathname } = useLocation();
  const isObservability = pathname === "/observability";
  const [cacheObservability, setCacheObservability] = useState(isObservability);
  const [token, setTokenState] = useState(getToken());

  useEffect(() => {
    if (isObservability) {
      setCacheObservability(true);
    }
  }, [isObservability]);

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
        {cacheObservability ? (
          <div className={isObservability ? undefined : "keep-alive-page"} aria-hidden={!isObservability}>
            <ObservabilityPage />
          </div>
        ) : null}
        {isObservability ? null : <Outlet />}
      </main>
    </div>
  );
}
