import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AccountsPage } from "./pages/Accounts";
import { AllocationsPage } from "./pages/Allocations";
import { CommandsPage } from "./pages/Commands";
import { ExchangesPage } from "./pages/Exchanges";
import { MarketFeedsPage } from "./pages/MarketFeeds";
import { OverviewPage } from "./pages/Overview";
import { PlatformSettingsPage } from "./pages/PlatformSettings";
import { PresetsPage } from "./pages/Presets";
import { StrategiesPage } from "./pages/Strategies";
import { WorkersPage } from "./pages/Workers";

export default function App() {
  return (
    <BrowserRouter basename="/ui">
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="exchanges" element={<ExchangesPage />} />
          <Route path="accounts" element={<AccountsPage />} />
          <Route path="presets" element={<PresetsPage />} />
          <Route path="feeds" element={<MarketFeedsPage />} />
          <Route path="strategies" element={<StrategiesPage />} />
          <Route path="allocations" element={<AllocationsPage />} />
          <Route path="workers" element={<WorkersPage />} />
          <Route path="settings" element={<PlatformSettingsPage />} />
          <Route path="commands" element={<CommandsPage />} />
          <Route path="observability" element={null} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
