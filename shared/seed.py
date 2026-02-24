from __future__ import annotations

import asyncio
import random

import asyncpg
from datasets import load_dataset

from shared.config import config
from shared.postgres import get_pool, init_schema, insert_product


SAMPLE_SIZE = 1000
random.seed(42)


def _map_row(item: dict) -> dict:
    return {
        "article_id": str(item["article_id"]),
        "name": item.get("prod_name", ""),
        "description": item.get("detail_desc") or None,
        "product_type": item.get("product_type_name") or None,
        "product_group": item.get("product_group_name") or None,
        "color": item.get("colour_group_name") or None,
        "department": item.get("department_name") or None,
        "index_name": item.get("index_name") or None,
        "image_url": item.get("image_url") or None,
        "price": round(random.uniform(9.99, 199.99), 2),
    }


async def seed() -> int:
    print("Initializing schema...")
    await init_schema()

    print(f"Loading {SAMPLE_SIZE} products from HuggingFace dataset...")
    ds = load_dataset("Qdrant/hm_ecommerce_products", split=f"train[:{SAMPLE_SIZE}]")

    pool = await get_pool()
    inserted = 0
    skipped = 0

    for item in ds:
        product = _map_row(item)
        try:
            await insert_product(product)
            inserted += 1
        except asyncpg.UniqueViolationError:
            skipped += 1

    print(f"Done. Inserted: {inserted}, Skipped (already exists): {skipped}")
    return inserted


if __name__ == "__main__":
    asyncio.run(seed())