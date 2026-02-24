from __future__ import annotations

from decimal import Decimal
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Document, PointStruct

from shared.config import config
from shared.embedding import article_id_to_uuid, build_embedding_text


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        kwargs: dict = {
            "url": config.qdrant_url,
            "timeout": 10,
            "cloud_inference": config.cloud_inference,
        }
        if config.qdrant_api_key:
            kwargs["api_key"] = config.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def init_collection() -> None:
    """
    Create the products collection if it doesn't exist.
    """
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if config.qdrant_collection not in existing:
        client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
    
    # create payload indices on commonly filtered fields for better performance
    for field in ["color", "product_type"]:
        try:
            client.create_payload_index(
                collection_name=config.qdrant_collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            print(f"Warning: Failed to create payload index on '{field}': {exc}")


def _product_to_point(product: dict) -> PointStruct:
    """
    Convert a product dict to a Qdrant PointStruct using Cloud Inference.
    """
    text = build_embedding_text(product)
    point_id = article_id_to_uuid(product["article_id"])

    # serialize Decimal to float for Qdrant payload
    payload = {
        k: float(v) if isinstance(v, Decimal) else v
        for k, v in product.items()
        if v is not None and k not in ("id", "created_at", "updated_at")
    }
    
    # keep datetime fields as ISO strings
    for dt_field in ("created_at", "updated_at"):
        if product.get(dt_field):
            payload[dt_field] = product[dt_field].isoformat() if hasattr(product[dt_field], "isoformat") else str(product[dt_field])

    return PointStruct(
        id=point_id,
        payload=payload,
        vector={
            "dense": Document(
                text=text,
                model=config.dense_model,
            ),
            "bm25": Document(
                text=text,
                model=config.sparse_model,
            ),
        },
    )


def upsert_product(product: dict) -> None:
    """
    Upsert a single product to Qdrant.
    """
    client = get_client()
    point = _product_to_point(product)
    client.upsert(
        collection_name=config.qdrant_collection,
        points=[point],
    )


def upsert_products_batch(products: list[dict], batch_size: int = 100) -> None:
    """
    Upsert a list of products in batches.
    """
    client = get_client()
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        points = [_product_to_point(p) for p in batch]
        client.upsert(
            collection_name=config.qdrant_collection,
            points=points,
        )


def delete_product(article_id: str) -> None:
    """
    Delete a product from Qdrant by article_id.
    """
    client = get_client()
    point_id = article_id_to_uuid(article_id)
    client.delete(
        collection_name=config.qdrant_collection,
        points_selector=models.PointIdsList(points=[point_id]),
    )


def get_all_point_ids() -> list[str]:
    """
    Scroll through all points and return their UUIDs mapped back to article_ids.
    """
    client = get_client()
    article_ids = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=config.qdrant_collection,
            limit=500,
            offset=offset,
            with_payload=["article_id"],
            with_vectors=False,
        )
        for point in results:
            if point.payload and "article_id" in point.payload:
                article_ids.append(point.payload["article_id"])
        if next_offset is None:
            break
        offset = next_offset

    return article_ids


def hybrid_search(
    query: str,
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Hybrid search combining dense + BM25 with RRF fusion.
    """
    client = get_client()

    filter_conditions = []
    if color:
        filter_conditions.append(
            models.FieldCondition(key="color", match=models.MatchValue(value=color))
        )
    if product_type:
        filter_conditions.append(
            models.FieldCondition(
                key="product_type", match=models.MatchValue(value=product_type)
            )
        )
    query_filter = (
        models.Filter(must=filter_conditions) if filter_conditions else None
    )

    results = client.query_points(
        collection_name=config.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=Document(text=query, model=config.dense_model),
                using="dense",
                limit=limit * 2,
                filter=query_filter,
            ),
            models.Prefetch(
                query=Document(text=query, model=config.sparse_model),
                using="bm25",
                limit=limit * 2,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    return [
        {**point.payload, "score": point.score}
        for point in results.points
    ]


def semantic_search(
    query: str,
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Dense-only semantic search.
    """
    client = get_client()

    filter_conditions = []
    if color:
        filter_conditions.append(
            models.FieldCondition(key="color", match=models.MatchValue(value=color))
        )
    if product_type:
        filter_conditions.append(
            models.FieldCondition(
                key="product_type", match=models.MatchValue(value=product_type)
            )
        )
    query_filter = (
        models.Filter(must=filter_conditions) if filter_conditions else None
    )

    results = client.query_points(
        collection_name=config.qdrant_collection,
        query=Document(text=query, model=config.dense_model),
        using="dense",
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {**point.payload, "score": point.score}
        for point in results.points
    ]


def keyword_search(
    query: str,
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    BM25-only keyword search.
    """
    client = get_client()

    filter_conditions = []
    if color:
        filter_conditions.append(
            models.FieldCondition(key="color", match=models.MatchValue(value=color))
        )
    if product_type:
        filter_conditions.append(
            models.FieldCondition(
                key="product_type", match=models.MatchValue(value=product_type)
            )
        )
    query_filter = (
        models.Filter(must=filter_conditions) if filter_conditions else None
    )

    results = client.query_points(
        collection_name=config.qdrant_collection,
        query=Document(text=query, model=config.sparse_model),
        using="bm25",
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {**point.payload, "score": point.score}
        for point in results.points
    ]


def check_health() -> bool:
    """
    Return True if Qdrant is reachable.
    """
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False