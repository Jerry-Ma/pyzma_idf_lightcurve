"""
DuckDB I/O Manager with JSON Type Handler for Dagster
Integrates with the official dagster-duckdb architecture while supporting JSON serialization.
"""

import json
import typing
from collections.abc import Sequence
from typing import Optional, Any, Dict, Union, Sequence

from dagster import InputContext, OutputContext, MetadataValue, io_manager
from dagster._core.storage.db_io_manager import DbTypeHandler, TableSlice, DbIOManager
from dagster_duckdb.io_manager import DuckDBIOManager, DuckDbClient
from loguru import logger

_PARTITION_EXPR = "partition_key"

class _DbIOManager(DbIOManager):
    """A DB I/O Manager that injects partition_expr metadata for partitioned assets."""

    def _inject_partition_expr(
        self, context: OutputContext
    ) -> None:
        """Inject partition_expr to context if is is partitioned."""
        if context.has_asset_partitions:
            if context.definition_metadata is None:
                context.definition_metadata = {}
            context.definition_metadata.setdefault("partition_expr", _PARTITION_EXPR)

    def handle_output(
        self, context: OutputContext, obj: object
    ) -> None:
        """Inject partition_expr to context if is is partitioned."""
        self._inject_partition_expr(context)
        super().handle_output(context, obj)

    def load_input(
        self, context: InputContext
    ) -> object:
        """Inject partition_expr to context if is is partitioned."""
        self._inject_partition_expr(context.upstream_output)
        return super().load_input(context)

    def _normalize_type(self, obj_type: type) -> type:
        obj_type_orig = typing.get_origin(obj_type)
        if obj_type_orig is not None:
            logger.debug(f"Using origin type {obj_type_orig} for {obj_type}")
            return obj_type_orig
        return obj_type 

    def _resolve_handler(self, obj_type: type) -> DbTypeHandler:
        return super()._resolve_handler(self._normalize_type(obj_type))

    def _check_supported_type(self, obj_type):
        return super()._check_supported_type(self._normalize_type(obj_type))

class DuckDBJSONTypeHandler(DbTypeHandler[Union[Dict, str, list, Any]]):
    """
    Stores and loads JSON-serializable objects (dicts, strings, lists, etc.) in DuckDB.
    """
    
    def handle_output(
        self, context: OutputContext, table_slice: TableSlice, obj: Any, connection
    ):
        """Stores the object as JSON in DuckDB."""
        # validate that only one partition is set in table_slice
        if table_slice.partition_dimensions and len(table_slice.partition_dimensions) > 1:
            raise ValueError("DuckDBJSONTypeHandler only supports single-partition assets.")
        if not table_slice.partition_dimensions or len(table_slice.partition_dimensions) == 0:
            partition_key = ""
        else:
            partition_keys = table_slice.partition_dimensions[0].partitions
            if len(partition_keys) != 1:
                raise ValueError("DuckDBJSONTypeHandler expects exactly one partition key.")
            partition_key = partition_keys[0]
        # Serialize the object to JSON
        value_json = json.dumps(obj, default=str)
        value_type = type(obj).__name__
 
        connection.execute(
            f"""create table if not exists {table_slice.schema}.{table_slice.table} (
                {_PARTITION_EXPR} VARCHAR UNIQUE,
                value_json VARCHAR,
                value_type VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # upsert the value
        connection.execute(f"""
            INSERT OR REPLACE INTO {table_slice.schema}.{table_slice.table}
            ({_PARTITION_EXPR}, value_json, value_type, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, [partition_key, value_json, value_type])
        logger.debug(f"✅ Stored {value_type} in DuckDB: {table_slice.schema}.{table_slice.table} partition {partition_key}")

        context.add_output_metadata(
            {
                "value_type": MetadataValue.text(value_type),
            }
        )

    def load_input(
        self, context: InputContext, table_slice: TableSlice, connection
    ) -> Any:
        """Loads the object from JSON in DuckDB."""
        if table_slice.partition_dimensions and len(context.asset_partition_keys) == 0:
            logger.debug("⚠️ Asset is partitioned but no partition key provided in context; cannot load partitioned data.")
            return None
        result = connection.execute(DuckDbClient.get_select_statement(table_slice)).fetchall()
        if len(result) != 1:
            logger.debug(f"⚠️ Expected exactly one result from DuckDB for {table_slice.schema}.{table_slice.table}, got {len(result)}")
            return None
        partition_key, value_json, value_type, created_at = result[0]
        # Deserialize from JSON
        obj = json.loads(value_json)
            
        logger.debug(f"✅ Loaded {value_type} from DuckDB: {table_slice.schema}.{table_slice.table} {partition_key} (created: {created_at})")
        return obj

    @property
    def supported_types(self):
        """Return the types this handler supports."""
        return [dict, str, list, tuple, int, float, bool, type(None)]


class DuckDBJSONIOManager(DuckDBIOManager):
    
    @classmethod
    def _is_dagster_maintained(cls) -> bool:
        return False  # This is our custom implementation
    
    @staticmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        return [DuckDBJSONTypeHandler()]
    
    @staticmethod
    def default_load_type() -> Optional[type]:
        return type(None)

    def create_io_manager(self, context) -> DbIOManager:
        return _DbIOManager(
            db_client=DuckDbClient(),
            database=self.database,
            schema=self.schema_,
            type_handlers=self.type_handlers(),
            default_load_type=self.default_load_type(),
            io_manager_name="DuckDBJSONIOManager",
        )

@io_manager(config_schema=DuckDBJSONIOManager.to_config_schema())
def duckdb_json_io_manager(init_context):
    return _DbIOManager(
        type_handlers=[DuckDBJSONTypeHandler()],
        db_client=DuckDbClient(),
        io_manager_name="DuckDBJSONIOManager",
        database=init_context.resource_config["database"],
        schema=init_context.resource_config.get("schema"),
        default_load_type=None,
    )