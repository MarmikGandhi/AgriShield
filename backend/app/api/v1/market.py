"""
market.py
=========
Mandi agricultural commodity price endpoints (APMC / Agmarknet / Data.gov.in).
"""

from typing import List, Optional
from fastapi import APIRouter, Query
from app.schemas.schemas import MandiPriceItem
from app.services.market_service import get_mandi_prices

router = APIRouter(prefix="/market", tags=["Mandi Prices"])

@router.get("/prices", response_model=List[MandiPriceItem])
def list_mandi_prices(
    commodity: Optional[str] = Query(None),
    district: Optional[str] = Query(None)
):
    prices = get_mandi_prices(commodity_filter=commodity, district_filter=district)
    return prices
