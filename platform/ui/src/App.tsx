import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AllocationsPage } from "./pages/Allocations";
import { CommandsPage } from "./pages/Commands";
import { ObservabilityPage } from "./pages/Observability";
import { OverviewPage } from "./pages/Overview";
import { StrategiesPage } from "./pages/Strategies";
import { WorkersPage } from "./pages/Workers";

export default function App() {
  return (
    <BrowserRouter basename="/ui">
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="strategies" element={<StrategiesPage />} />
          <Route path="allocations" element={<AllocationsPage />} />
          <Route path="workers" element={<WorkersPage />} />
          <Route path="commands" element={<CommandsPage />} />
          <Route path="observability" element={<ObservabilityPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
