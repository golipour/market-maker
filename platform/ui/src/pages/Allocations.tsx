import { Navigate } from "react-router-dom";

/** Legacy route — capital pools live on the Strategies page. */
export function AllocationsPage() {
  return <Navigate to="/strategies#capital-pools" replace />;
}
