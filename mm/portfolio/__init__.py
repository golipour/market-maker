"""Portfolio virtual ledgers and capital attribution."""

from mm.portfolio.virtual_ledger_service import VirtualLedgerService

# Backward-compatible alias.
PortfolioAllocationService = VirtualLedgerService

__all__ = ["VirtualLedgerService", "PortfolioAllocationService"]
