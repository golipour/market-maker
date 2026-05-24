"""Quick check: binding uses config local; get_exchange_info uses the exchange argument."""

from mm.runtime.config import load_runtime_config
from mm.integrations.trade_stats import get_exchange_info

cfg = load_runtime_config("mm/configs/market_maker.runtime.example.json")

print("binding.local (from config venues.local=nobitex):")
print(cfg.binding.local)

print("\nget_exchange_info(..., 'nobitex'):")
print(get_exchange_info("USDT_IRT", "nobitex"))

print("\nget_exchange_info(..., 'wallex'):")
print(get_exchange_info("USDT_IRT", "wallex"))

if cfg.global_ is not None:
    print("\nbinding.global (from config venues.global):")
    print(cfg.binding.global_)
