"""Preview PostgreSQL table DDL; use --apply only for initial/missing tables."""
import argparse
import asyncio
import importlib
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable
from src.core.base_model import Base
from src.infra.database import engine
from src.modules.medical import model as medical_model


def load_models():
    try:
        importlib.import_module("src.agents.knowledge.model")
    except ModuleNotFoundError as exc:
        if exc.name not in {"src.agents.knowledge", "src.agents.knowledge.model"}:
            raise


def preview():
    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        print(str(CreateTable(table).compile(dialect=dialect)).strip() + ";")
        for index in sorted(table.indexes, key=lambda item: item.name):
            print(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";")


async def apply():
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create missing tables; does not alter existing columns")
    args = parser.parse_args()
    load_models()
    if args.apply:
        asyncio.run(apply())
    else:
        preview()


if __name__ == "__main__":
    main()
