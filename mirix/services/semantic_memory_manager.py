import uuid
import random
import string
import time
from typing import List, Optional, Dict, Any
import json
import re

import numpy as np
from mirix.log import get_logger

logger = get_logger(__name__)
from mirix.orm.errors import NoResultFound
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.schemas.user import User as PydanticUser
from mirix.schemas.semantic_memory import (
    SemanticMemoryItem as PydanticSemanticMemoryItem,
    SemanticMemoryItemUpdate
)
from mirix.utils import enforce_types, generate_short_id, generate_unique_short_id
from pydantic import BaseModel
from sqlalchemy import select, func, text
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

from mirix.schemas.agent import AgentState
from mirix.embeddings import embedding_model, parse_and_chunk_text
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.services.utils import build_query, update_timezone
from mirix.settings import settings
from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

class SemanticMemoryManager:
    """Manager class to handle business logic related to Semantic Memory Items."""

    def __init__(self):
        from mirix.server.server import db_context
        self.session_maker = db_context

    def _clean_text_for_search(self, text: str) -> str:
        """
        Clean text by removing punctuation and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text with punctuation removed and normalized whitespace
        """
        if not text:
            return ""
        
        # Remove punctuation using string.punctuation
        # Create translation table that maps each punctuation character to space
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        return text

    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 search by tokenizing and cleaning.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of cleaned tokens
        """
        if not text:
            return []
        
        # Clean text first
        cleaned_text = self._clean_text_for_search(text)
        
        # Split into tokens and filter out empty strings and very short tokens
        tokens = [token for token in cleaned_text.split() if token.strip() and len(token) > 1]
        return tokens

    def _parse_embedding_field(self, embedding_value):
        """
        Helper method to parse embedding field from different PostgreSQL return formats.
        
        Args:
            embedding_value: The raw embedding value from PostgreSQL query
            
        Returns:
            List of floats or None if parsing fails
        """
        if embedding_value is None:
            return None
        
        try:
            # If it's already a list or tuple, convert to list
            if isinstance(embedding_value, (list, tuple)):
                return list(embedding_value)
            
            # If it's a string, try different parsing approaches
            if isinstance(embedding_value, str):
                # Remove any whitespace
                embedding_value = embedding_value.strip()
                
                # Check if it's a JSON array string: "[-0.006639634,-0.0114432...]"
                if embedding_value.startswith('[') and embedding_value.endswith(']'):
                    try:
                        return json.loads(embedding_value)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try manual parsing
                        # Remove brackets and split by comma
                        inner = embedding_value[1:-1]  # Remove [ and ]
                        return [float(x.strip()) for x in inner.split(',') if x.strip()]
                
                # Try comma-separated values
                if ',' in embedding_value:
                    return [float(x.strip()) for x in embedding_value.split(',') if x.strip()]
                
                # Try space-separated values
                if ' ' in embedding_value:
                    return [float(x.strip()) for x in embedding_value.split() if x.strip()]
            
            # Try using the original deserialize_vector approach for binary data
            try:
                from mirix.helpers.converters import deserialize_vector
                class MockDialect:
                    name = 'postgresql'
                return deserialize_vector(embedding_value, MockDialect())
            except Exception:
                pass
                
            # If all else fails, return None to avoid validation errors
            return None
            
        except Exception as e:
            print(f"Warning: Failed to parse embedding field: {e}")
            return None

    def _count_word_matches(self, item_data: Dict[str, Any], query_words: List[str], search_field: str = '') -> int:
        """
        Count how many of the query words are present in the semantic memory item data.
        
        Args:
            item_data: Dictionary containing semantic memory item data
            query_words: List of query words to search for
            search_field: Specific field to search in, or empty string to search all text fields
            
        Returns:
            Number of query words found in the item
        """
        if not query_words:
            return 0
        
        # Determine which text fields to search in
        if search_field == 'name':
            search_texts = [item_data.get('name', '')]
        elif search_field == 'summary':
            search_texts = [item_data.get('summary', '')]
        elif search_field == 'details':
            search_texts = [item_data.get('details', '')]
        elif search_field == 'source':
            search_texts = [item_data.get('source', '')]
        else:
            # Search across all relevant text fields
            search_texts = [
                item_data.get('name', ''),
                item_data.get('summary', ''),
                item_data.get('details', ''),
                item_data.get('source', '')
            ]
        
        # Combine all search texts and clean them (remove punctuation)
        combined_text = ' '.join(text for text in search_texts if text)
        cleaned_combined_text = self._clean_text_for_search(combined_text)
        
        # Count how many query words are present
        word_matches = 0
        for word in query_words:
            # Query words are already cleaned, so we can do direct comparison
            if word in cleaned_combined_text:
                word_matches += 1
        
        return word_matches

    def _postgresql_fulltext_search(self, session, base_query, query_text, search_field, limit):
        """
        Efficient PostgreSQL-native full-text search using ts_rank_cd for BM25-like functionality.
        This method leverages PostgreSQL's built-in full-text search capabilities and GIN indexes.
        
        Args:
            session: Database session
            base_query: Base SQLAlchemy query
            query_text: Search query string
            search_field: Field to search in ('name', 'summary', 'details', 'source', etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of SemanticMemoryItem objects ranked by relevance
        """
        from sqlalchemy import text, func
        
        # Clean and prepare the search query
        cleaned_query = self._clean_text_for_search(query_text)
        if not cleaned_query.strip():
            return []
        
        # Split into words and create a tsquery - PostgreSQL will handle the ranking
        query_words = [word.strip() for word in cleaned_query.split() if word.strip()]
        if not query_words:
            return []
        
        # Create tsquery string with improved logic
        tsquery_parts = []
        for word in query_words:
            # Escape special characters for tsquery
            escaped_word = word.replace("'", "''").replace("&", "").replace("|", "").replace("!", "").replace(":", "")
            if escaped_word and len(escaped_word) > 1:  # Skip very short words
                # Add both exact and prefix matching for better results
                if len(escaped_word) >= 3:
                    tsquery_parts.append(f"('{escaped_word}' | '{escaped_word}':*)")
                else:
                    tsquery_parts.append(f"'{escaped_word}'")
        
        if not tsquery_parts:
            return []
        
        # Use AND logic for multiple terms to find more relevant documents
        # but fallback to OR if AND produces no results
        if len(tsquery_parts) > 1:
            tsquery_string_and = " & ".join(tsquery_parts)  # AND logic for precision
            tsquery_string_or = " | ".join(tsquery_parts)   # OR logic for recall
        else:
            tsquery_string_and = tsquery_string_or = tsquery_parts[0]
        
        # Determine which field to search based on search_field
        if search_field == 'name':
            tsvector_sql = "to_tsvector('english', coalesce(name, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(name, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'summary':
            tsvector_sql = "to_tsvector('english', coalesce(summary, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(summary, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'details':
            tsvector_sql = "to_tsvector('english', coalesce(details, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(details, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'source':
            tsvector_sql = "to_tsvector('english', coalesce(source, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(source, '')), to_tsquery('english', :tsquery), 32)"
        else:
            # Search across all relevant text fields with weighting
            tsvector_sql = """setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
                             setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                             setweight(to_tsvector('english', coalesce(details, '')), 'C') ||
                             setweight(to_tsvector('english', coalesce(source, '')), 'D')"""
            rank_sql = f"""ts_rank_cd(
                setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(details, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(source, '')), 'D'),
                to_tsquery('english', :tsquery), 32)"""
        
        # Try AND query first for more precise results
        try:
            and_query_sql = text(f"""
                SELECT 
                    id, created_at, name, summary, details, source, tree_path,
                    name_embedding, summary_embedding, details_embedding, embedding_config,
                    organization_id, metadata_, last_modify,
                    {rank_sql} as rank_score
                FROM semantic_memory 
                WHERE {tsvector_sql} @@ to_tsquery('english', :tsquery)
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """)
            
            results = list(session.execute(and_query_sql, {
                'tsquery': tsquery_string_and,
                'limit_val': limit or 50
            }))
            
            # If AND query returns sufficient results, use them
            if len(results) >= min(limit or 10, 10):
                semantic_items = []
                for row in results:
                    data = dict(row._mapping)
                    # Remove the rank_score field before creating the object
                    data.pop('rank_score', None)
                    
                    # Parse JSON fields that are returned as strings from raw SQL
                    json_fields = ['last_modify', 'metadata_', 'embedding_config']
                    for field in json_fields:
                        if field in data and isinstance(data[field], str):
                            try:
                                data[field] = json.loads(data[field])
                            except (json.JSONDecodeError, TypeError):
                                pass
                    
                    # Parse embedding fields
                    embedding_fields = ['name_embedding', 'summary_embedding', 'details_embedding']
                    for field in embedding_fields:
                        if field in data and data[field] is not None:
                            data[field] = self._parse_embedding_field(data[field])
                    
                    semantic_items.append(SemanticMemoryItem(**data))
                
                return [item.to_pydantic() for item in semantic_items]
                
        except Exception as e:
            print(f"PostgreSQL AND query error: {e}")
        
        # If AND query fails or returns too few results, try OR query
        try:
            or_query_sql = text(f"""
                SELECT 
                    id, created_at, name, summary, details, source, tree_path,
                    name_embedding, summary_embedding, details_embedding, embedding_config,
                    organization_id, metadata_, last_modify,
                    {rank_sql} as rank_score
                FROM semantic_memory 
                WHERE {tsvector_sql} @@ to_tsquery('english', :tsquery)
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """)
            
            results = session.execute(or_query_sql, {
                'tsquery': tsquery_string_or,
                'limit_val': limit or 50
            })
            
            semantic_items = []
            for row in results:
                data = dict(row._mapping)
                # Remove the rank_score field before creating the object
                data.pop('rank_score', None)
                
                # Parse JSON fields that are returned as strings from raw SQL
                json_fields = ['last_modify', 'metadata_', 'embedding_config']
                for field in json_fields:
                    if field in data and isinstance(data[field], str):
                        try:
                            data[field] = json.loads(data[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                # Parse embedding fields
                embedding_fields = ['name_embedding', 'summary_embedding', 'details_embedding']
                for field in embedding_fields:
                    if field in data and data[field] is not None:
                        data[field] = self._parse_embedding_field(data[field])
                
                semantic_items.append(SemanticMemoryItem(**data))
            
            return [item.to_pydantic() for item in semantic_items]
            
        except Exception as e:
            # If there's an error with the tsquery, fall back to simpler search
            print(f"PostgreSQL full-text search error: {e}")
            # Fall back to simple ILIKE search
            fallback_field = getattr(SemanticMemoryItem, search_field) if search_field and hasattr(SemanticMemoryItem, search_field) else SemanticMemoryItem.name
            fallback_query = base_query.where(
                func.lower(fallback_field).contains(query_text.lower())
            ).order_by(SemanticMemoryItem.created_at.desc())
            
            if limit:
                fallback_query = fallback_query.limit(limit)
                
            results = session.execute(fallback_query)
            semantic_items = [SemanticMemoryItem(**dict(row._mapping)) for row in results]
            return [item.to_pydantic() for item in semantic_items]

    @update_timezone
    @enforce_types
    def get_semantic_item_by_id(self, semantic_memory_id: str, timezone_str: str) -> Optional[PydanticSemanticMemoryItem]:
        """Fetch a semantic memory item by ID."""
        with self.session_maker() as session:
            try:
                semantic_memory_item = SemanticMemoryItem.read(db_session=session, identifier=semantic_memory_id)
                return semantic_memory_item.to_pydantic()
            except NoResultFound:
                raise NoResultFound(f"Semantic memory item with id {semantic_memory_id} not found.")

    @update_timezone
    @enforce_types
    def get_most_recently_updated_item(self, organization_id: Optional[str] = None, timezone_str: str = None) -> Optional[PydanticSemanticMemoryItem]:
        """
        Fetch the most recently updated semantic memory item based on last_modify timestamp.
        Optionally filter by organization_id.
        Returns None if no items exist.
        """
        with self.session_maker() as session:
            # Use proper PostgreSQL JSON text extraction and casting for ordering
            from sqlalchemy import cast, DateTime, text
            query = select(SemanticMemoryItem).order_by(
                cast(text("semantic_memory.last_modify ->> 'timestamp'"), DateTime).desc()
            )
            
            if organization_id:
                query = query.where(SemanticMemoryItem.organization_id == organization_id)
            
            result = session.execute(query.limit(1))
            item = result.scalar_one_or_none()
            
            return [item.to_pydantic()] if item else None

    @enforce_types
    def create_item(self, item_data: PydanticSemanticMemoryItem) -> PydanticSemanticMemoryItem:
        """Create a new semantic memory item."""
        
        # Ensure ID is set before model_dump
        if not item_data.id:
            item_data.id = generate_unique_short_id(self.session_maker, SemanticMemoryItem, "sem")
        
        data_dict = item_data.model_dump()

        # Validate required fields
        required_fields = ["summary", "name"]
        for field in required_fields:
            if field not in data_dict or not data_dict[field]:
                raise ValueError(f"Required field '{field}' missing from semantic memory data")
        
        data_dict.setdefault("metadata_", {})

        with self.session_maker() as session:
            item = SemanticMemoryItem(**data_dict)
            item.create(session)
            return item.to_pydantic()

    @enforce_types
    def update_item(self, item_update: SemanticMemoryItemUpdate, actor: PydanticUser) -> PydanticSemanticMemoryItem:
        """Update an existing semantic memory item."""
        with self.session_maker() as session:
            item = SemanticMemoryItem.read(db_session=session, identifier=item_update.id, actor=actor)
            update_data = item_update.model_dump(exclude_unset=True)
            for k, v in update_data.items():
                if k not in ["id", "updated_at"]:
                    setattr(item, k, v)
            item.updated_at = item_update.updated_at
            item.update(session, actor=actor)
            return item.to_pydantic()

    @enforce_types
    def create_many_items(self, items: List[PydanticSemanticMemoryItem], actor: PydanticUser) -> List[PydanticSemanticMemoryItem]:
        """Create multiple semantic memory items."""
        return [self.create_item(i) for i in items]

    def get_total_number_of_items(self) -> int:
        """Get the total number of items in the semantic memory."""
        with self.session_maker() as session:
            query = select(func.count(SemanticMemoryItem.id))
            result = session.execute(query)
            return result.scalar_one()

    @update_timezone
    @enforce_types
    def list_semantic_items(self, 
                            agent_state: AgentState,
                            query: str = '', 
                            embedded_text: Optional[List[float]] = None,
                            search_field: str = '',
                            search_method: str = 'embedding',
                            limit: Optional[int] = 50,
                            timezone_str: str = None) -> List[PydanticSemanticMemoryItem]:
        """
        List semantic memory items with various search methods.
        
        Args:
            agent_state: The agent state containing embedding configuration
            query: Search query string
            embedded_text: Pre-computed embedding for semantic search
            search_field: Field to search in ('name', 'summary', 'details', 'source')
            search_method: Search method to use:
                - 'embedding': Vector similarity search using embeddings
                - 'string_match': Simple string containment search
                - 'bm25': **RECOMMENDED** - PostgreSQL native full-text search (ts_rank_cd) when using PostgreSQL, 
                               falls back to in-memory BM25 for SQLite
                - 'fuzzy_match': Fuzzy string matching (legacy, kept for compatibility)
            limit: Maximum number of results to return
            timezone_str: Timezone string for timestamp conversion
            
        Returns:
            List of semantic memory items matching the search criteria
            
        Note:
            **For PostgreSQL users**: 'bm25' is now the recommended method for text-based searches as it uses 
            PostgreSQL's native full-text search with ts_rank_cd for BM25-like scoring. This is much more efficient 
            than loading all documents into memory and leverages your existing GIN indexes.
            
            **For SQLite users**: 'bm25' now has fallback support that uses in-memory BM25 processing.
            
            Performance comparison:
            - PostgreSQL 'bm25': Native DB search, very fast, scales well
            - Fallback 'bm25' (SQLite): In-memory processing, slower for large datasets but still provides 
              proper BM25 ranking
        """
        with self.session_maker() as session:
            
            if query == '':
                # Use proper PostgreSQL JSON text extraction and casting for ordering
                from sqlalchemy import cast, DateTime, text
                query_stmt = select(SemanticMemoryItem).order_by(
                    cast(text("semantic_memory.last_modify ->> 'timestamp'"), DateTime).desc()
                )
                if limit:
                    query_stmt = query_stmt.limit(limit)
                result = session.execute(query_stmt)
                semantic_items = result.scalars().all()
                return [item.to_pydantic() for item in semantic_items]

            else:
                
                base_query = select(
                    SemanticMemoryItem.id.label("id"),
                    SemanticMemoryItem.created_at.label("created_at"),
                    SemanticMemoryItem.name.label("name"),
                    SemanticMemoryItem.summary.label("summary"),
                    SemanticMemoryItem.details.label("details"),
                    SemanticMemoryItem.source.label("source"),
                    SemanticMemoryItem.name_embedding.label("name_embedding"),
                    SemanticMemoryItem.summary_embedding.label("summary_embedding"),
                    SemanticMemoryItem.details_embedding.label("details_embedding"),
                    SemanticMemoryItem.embedding_config.label("embedding_config"),
                    SemanticMemoryItem.organization_id.label("organization_id"),
                    SemanticMemoryItem.metadata_.label("metadata_"),
                    SemanticMemoryItem.last_modify.label("last_modify"),
                    SemanticMemoryItem.tree_path.label("tree_path"),
                )

                if search_method == 'embedding':
                    embed_query = True
                    embedding_config = agent_state.embedding_config

                    main_query = build_query(
                        base_query=base_query,
                        query_text=query,
                        embedded_text=embedded_text,
                        embed_query=embed_query,
                        embedding_config=embedding_config,
                        search_field=eval("SemanticMemoryItem." + search_field + "_embedding"),
                        target_class=SemanticMemoryItem,
                    )

                elif search_method == 'string_match':

                    search_field = eval("SemanticMemoryItem." + search_field)
                    main_query = base_query.where(func.lower(search_field).contains(query.lower()))

                elif search_method == 'bm25':
                    
                    # Check if we're using PostgreSQL - use native full-text search if available
                    if settings.mirix_pg_uri_no_default:
                        # Use PostgreSQL native full-text search
                        return self._postgresql_fulltext_search(
                            session, base_query, query, search_field, limit
                        )
                    else:
                        # Fallback to in-memory BM25 for SQLite (legacy method)
                        # Load all candidate items (memory-intensive, kept for compatibility)
                        result = session.execute(select(SemanticMemoryItem))
                        all_items = result.scalars().all()
                        
                        if not all_items:
                            return []
                        
                        # Prepare documents for BM25
                        documents = []
                        valid_items = []
                        
                        for item in all_items:
                            # Determine which field to use for search
                            if search_field and hasattr(item, search_field):
                                text_to_search = getattr(item, search_field) or ""
                            else:
                                text_to_search = item.name or ""
                            
                            # Preprocess the text into tokens
                            tokens = self._preprocess_text_for_bm25(text_to_search)
                            
                            # Only include items that have tokens after preprocessing
                            if tokens:
                                documents.append(tokens)
                                valid_items.append(item)
                        
                        if not documents:
                            return []
                        
                        # Initialize BM25 with the documents
                        bm25 = BM25Okapi(documents)
                        
                        # Preprocess the query
                        query_tokens = self._preprocess_text_for_bm25(query)
                        
                        if not query_tokens:
                            # If query has no valid tokens, return most recent items
                            return [item.to_pydantic() for item in valid_items[:limit]]
                        
                        # Get BM25 scores for all documents
                        scores = bm25.get_scores(query_tokens)
                        
                        # Create scored items list
                        scored_items = list(zip(scores, valid_items))
                        
                        # Sort by BM25 score in descending order
                        scored_items.sort(key=lambda x: x[0], reverse=True)
                        
                        # Get top items based on limit
                        top_items = [item for score, item in scored_items[:limit]]
                        semantic_items = top_items
                        
                        # Return the list after converting to Pydantic
                        return [item.to_pydantic() for item in semantic_items]

                elif search_method == 'fuzzy_match':
                    # Fuzzy matching: load all candidate items into memory and compute a fuzzy match score.
                    result = session.execute(select(SemanticMemoryItem))
                    all_items = result.scalars().all()
                    scored_items = []
                    for item in all_items:
                        # Determine which field to use:
                        # 1. If a search_field is provided (e.g., "concept" or "summary") and exists in the item, use it.
                        # 2. Otherwise, default to using the "concept" field.
                        if search_field and hasattr(item, search_field):
                            text_to_search = getattr(item, search_field)
                        else:
                            text_to_search = item.name
                        # Compute the fuzzy matching score using partial_ratio for better short-to-long matching.
                        score = fuzz.partial_ratio(query.lower(), text_to_search.lower())
                        scored_items.append((score, item))
                    
                    # Sort items descending by score and pick the top ones.
                    scored_items.sort(key=lambda x: x[0], reverse=True)
                    top_items = [item for score, item in scored_items[:limit]]
                    return [item.to_pydantic() for item in top_items]

                if limit:
                    main_query = main_query.limit(limit)

                results = list(session.execute(main_query))

                semantic_items = []
                for row in results:
                    data = dict(row._mapping)
                    semantic_items.append(SemanticMemoryItem(**data))

                return [item.to_pydantic() for item in semantic_items]

    @enforce_types
    def insert_semantic_item(
        self,
        agent_state: AgentState,
        name: str,
        summary: str,
        details: Optional[str],
        source: Optional[str],
        tree_path: Optional[List[str]],
        organization_id: str
    ) -> PydanticSemanticMemoryItem:
        """
        Create a new semantic memory entry using provided parameters.
        """
        try:

            # Conditionally calculate embeddings based on BUILD_EMBEDDINGS_FOR_MEMORY flag
            if BUILD_EMBEDDINGS_FOR_MEMORY:
                # TODO: need to check if we need to chunk the text
                embed_model = embedding_model(agent_state.embedding_config)
                name_embedding = embed_model.get_text_embedding(name)
                summary_embedding = embed_model.get_text_embedding(summary)
                details_embedding = embed_model.get_text_embedding(details)
                embedding_config = agent_state.embedding_config
            else:
                name_embedding = None
                summary_embedding = None
                details_embedding = None
                embedding_config = None

            semantic_item = self.create_item(
                item_data=PydanticSemanticMemoryItem(
                    name=name,
                    summary=summary,
                    details=details,
                    source=source,
                    organization_id=organization_id,
                    details_embedding=details_embedding,
                    name_embedding=name_embedding,
                    summary_embedding=summary_embedding,
                    embedding_config=embedding_config,
                    tree_path=tree_path,
                )
            )
            
            # Note: Item is already added to clustering tree in create_item()
            return semantic_item
        except Exception as e:
            raise e


    def delete_semantic_item_by_id(self, semantic_memory_id: str) -> None:
        """Delete a semantic memory item by ID."""
        with self.session_maker() as session:
            try:
                item = SemanticMemoryItem.read(db_session=session, identifier=semantic_memory_id)
                item.hard_delete(session)
            except NoResultFound:
                raise NoResultFound(f"Semantic memory item with id {semantic_memory_id} not found.")
